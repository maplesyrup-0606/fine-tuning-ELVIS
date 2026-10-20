"""
Per-principle, per-generator ground-truth rubric specs, derived from ELVIS's generators.

A PrincipleSpec maps each ELVIS generator to a GeneratorSpec holding everything
generator-specific:
  - props:            the properties the generator varies (rel/irrel/free come from these)
  - grouping:         slots -> the `Grouping:` sentence
  - relevant:         slots -> the `Relevant:` sentence
  - irrelevant_scope: how irrelevant props are held fixed: "image" (one value per image),
                      "cluster" (one value per cluster) or "free" (not applied at all)
  - violations:       (objs, slots, color_names) -> [(category, evidence)]; [] = meets the rule
  - grouping_checked: whether `violations` can detect a broken grouping from metadata

Shared: pattern-name parsing (all three ELVIS name formats), rendering, the irrelevant
check. generate_programmatic_rules.py, check_ft_data.py and show_eval_episode.py read
these definitions. Survey + decisions: EXPERIMENT.md "v3 generator survey decisions".

Rubric lines: Principle / Grouping / Relevant / Irrelevant, optionally followed by
`Image 4-6` lines describing an episode's example negatives (v2 only).
"""

import math
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional

NUMBER_WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six"}

# ELVIS task names, prefixed with main.py's folder index. Three formats:
#   create_tasks_v3: {idx}_{n}_{generator}_rel_{props}_{groups}_{quantity}_irrel_{props}
#   create_tasks_v4: same, plus _{all|exist}
#   create_tasks_v2: no _{quantity}
NAME_RE = re.compile(
    r"^\d+_\d+_(?P<generator>.+?)_rel_(?P<rel>.+?)_(?P<groups>\d+)(?:_(?P<quantity>[a-z0-9]+))?"
    r"_irrel_(?P<irrel>.+?)(?:_(?P<qualifier>all|exist))?$"
)

IRRELEVANT_TEXT = {
    "image": "same for every object in an image",
    "cluster": "same within each cluster",
    "free": "varies freely",
}


@dataclass
class GeneratorSpec:
    props: tuple
    grouping: Callable[[dict], str]
    relevant: Callable[[dict], str]
    violations: Callable[[list, dict, list], list]
    irrelevant_scope: str = "image"
    grouping_checked: bool = True
    # props left out of the Irrelevant line unless relevant (e.g. palette count: always
    # equal unless deliberately broken, so "varies freely" would be false)
    omit_unless_relevant: tuple = ()
    # prop -> wording replacing the scope text for an irrelevant prop (e.g. count)
    irrelevant_text: dict = field(default_factory=dict)
    # (objs, color_names) -> the objects the rubric is about (drops invisible helper objects,
    # e.g. feature_circle's background-colored occluder)
    objects: Callable[[list, list], list] = field(default=lambda objs, color_names: objs)


@dataclass
class PrincipleSpec:
    name: str
    generators: Dict[str, GeneratorSpec]
    expected_patterns: Optional[int] = None
    # slots -> True if the pattern must not be used for training (known-bad labels)
    exclude_from_training: Callable[[dict], bool] = field(default=lambda slots: False)

    def parse(self, name):
        m = NAME_RE.match(name)
        if not m or m["generator"] not in self.generators:
            raise ValueError(f"unrecognized {self.name} pattern name: {name}")
        gen = self.generators[m["generator"]]
        rel, irrel = m["rel"].split("_"), m["irrel"].split("_")
        unknown = [p for p in rel + irrel if p not in gen.props]
        if unknown:
            raise ValueError(f"unknown properties {unknown} in pattern name: {name}")
        return {
            "generator": m["generator"],
            "groups": int(m["groups"]),
            "quantity": m["quantity"],
            "qualifier": m["qualifier"],
            "relevant": rel,
            "irrelevant": irrel,
            "free": [p for p in gen.props if p not in rel and p not in irrel],
        }

    def violations(self, objs, slots, color_names):
        """Ordered list of (category, evidence) the image breaks; [] = meets the rule."""
        gen = self.generators[slots["generator"]]
        return gen.violations(gen.objects(objs, color_names), slots, color_names)

    def irrelevant_errors(self, objs, slots, color_names):
        """Irrelevant props that are not held fixed the way this generator promises."""
        gen = self.generators[slots["generator"]]
        objs = gen.objects(objs, color_names)
        scope = gen.irrelevant_scope
        if scope == "free":
            return []
        groups = clusters(objs) if scope == "cluster" else {0: foreground(objs)}
        errors = []
        for p in slots["irrelevant"]:
            if p not in ("shape", "color", "size"):
                continue   # count/axis: not a per-object value
            if any(len(set(values(g, p, color_names))) > 1 for g in groups.values()):
                errors.append(f"irrel {p} not constant per {scope}")
        return errors

    def render(self, slots, neg_violations=(), first_neg_index=4):
        gen = self.generators[slots["generator"]]
        scope_text = IRRELEVANT_TEXT[gen.irrelevant_scope]
        shown = lambda props: [p for p in props if p not in gen.omit_unless_relevant]
        irrelevant = ([f"{p} ({gen.irrelevant_text.get(p, scope_text)})" for p in shown(slots["irrelevant"])]
                      + [f"{p} ({IRRELEVANT_TEXT['free']})" for p in shown(slots["free"])])
        lines = [
            f"Principle: {self.name}",
            f"Grouping: {gen.grouping(slots)}",
            f"Relevant: {gen.relevant(slots)}",
            f"Irrelevant: {'; '.join(irrelevant)}",
        ]
        for i, vs in enumerate(neg_violations):
            lines.append(f"Image {first_neg_index + i}: {'; '.join(f'{c} ({e})' for c, e in vs)}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# metadata helpers
# ---------------------------------------------------------------------------
def foreground(objs):
    """Objects that belong to a group (drops background objects stored with group_id -1)."""
    return [o for o in objs if o["group_id"] >= 0]


def clusters(objs):
    g = defaultdict(list)
    for o in foreground(objs):
        g[o["group_id"]].append(o)
    return dict(g)


def values(objs, prop, color_names):
    if prop == "shape":
        return [o["shape"] for o in objs]
    if prop == "color":
        return [color_names[o["color_name"]] for o in objs]
    if prop == "size":
        return [round(o["size"], 6) for o in objs]
    raise ValueError(f"no metadata accessor for property {prop!r}")


def obj_props(objs, color_names):
    return {p: values(objs, p, color_names) for p in ("shape", "color", "size")}


def _join(items, last="and"):
    items = list(items)
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + f" {last} " + items[-1]


def restricted_values(allowed, relevant_text):
    """Relevant line + violation check for 'each relevant prop restricted to a value list'
    (size: one fixed size). Used by continuity, grid and the other value-list generators."""
    def relevant(slots):
        return "; ".join(relevant_text[p] for p in slots["relevant"])

    def check(objs, slots, color_names, pool=None):
        """pool: objects whose props are checked (default: drop group_id -1 background)."""
        out = []
        pool = foreground(objs) if pool is None else pool
        for prop in slots["relevant"]:
            vals = values(pool, prop, color_names)
            if prop == "size":
                if len(set(vals)) > 1:
                    out.append(("size", "objects differ in size"))
            elif not set(vals) <= allowed[prop]:
                if prop == "shape":
                    off = sorted(set(vals) - allowed["shape"])
                    out.append(("shape", f"has {', '.join(off[:3])}"))
                else:
                    out.append(("color", f"has colors other than {' and '.join(sorted(allowed['color']))}"))
        return out
    return relevant, check


# ---------------------------------------------------------------------------
# continuity — ELVIS scripts/continuity/util_*.py
#   positive: objects on the generator's layout (group_id >= 0) and every rel prop
#   within the shared logic dict {shape: square/circle, color: green/yellow, size: fixed}.
#   negative: dropping "continuity" scatters the objects randomly (group_id = -1).
# ---------------------------------------------------------------------------
_cont_relevant, _cont_check = restricted_values(
    {"shape": {"square", "circle"}, "color": {"green", "yellow"}},
    {"shape": "shape (only square and circle)",
     "color": "color (only green and yellow)",
     "size": "size (all objects the same size)"},
)


def _cont_violations(objs, slots, color_names):
    # Scattered continuity objects carry group_id -1, so check props on every object.
    out = [("scattered", "objects do not follow the lines")] if any(o["group_id"] < 0 for o in objs) else []
    return out + _cont_check(objs, slots, color_names, pool=objs)


def _continuity(sentence):
    return GeneratorSpec(props=("shape", "color", "size"), grouping=lambda s: sentence(s["groups"]),
                         relevant=_cont_relevant, violations=_cont_violations)


CONTINUITY = PrincipleSpec(
    name="continuity",
    generators={
        "non_overlap_one_split_n": _continuity(lambda g: (
            f"objects are evenly spaced along one straight main line that splits into "
            f"{NUMBER_WORDS[g]} straight branches")),
        "with_intersected_n_splines": _continuity(lambda g: (
            f"objects are evenly spaced along {NUMBER_WORDS[g]} smooth curves that cross at one shared point")),
        "non_intersected_n_splines": _continuity(lambda g: (
            "objects are evenly spaced along one smooth curve" if g == 1 else
            f"objects are evenly spaced along {NUMBER_WORDS[g]} separate smooth curves that do not cross")),
        "feature_continuity_overlap_splines": _continuity(lambda g: (
            f"objects are evenly spaced along {NUMBER_WORDS[g]} smooth curves; the first curve has a "
            f"second row of objects running closely parallel to it")),
    },
    expected_patterns=432,
)


# ---------------------------------------------------------------------------
# proximity — ELVIS scripts/proximity/util_*.py (red_triangle size bug fixed in ELVIS)
#   Broken clusters keep their group ids (objects are just placed randomly), so
#   grouping is only checkable from metadata for circle_features (cluster count).
# ---------------------------------------------------------------------------
_RT_FEATURE = {
    "shape": "consists only of triangles",
    "color": "contains at least one red object",
    "size": "has all its objects the same size",
}


def _rt_has_feature(cluster, relevant, color_names):
    ok = True
    if "shape" in relevant:
        ok &= set(values(cluster, "shape", color_names)) == {"triangle"}
    if "color" in relevant:
        ok &= "red" in values(cluster, "color", color_names)
    if "size" in relevant:
        ok &= len(set(values(cluster, "size", color_names))) == 1
    return ok


def _rt_relevant(slots):
    feature = _join(_RT_FEATURE[p] for p in slots["relevant"])
    if slots["qualifier"] == "all":
        return f"every cluster {feature}"
    # Not "but not every": ELVIS applies the kept-props subset to positives too, so an
    # `exist` positive can have the feature in every cluster (45% in a sample).
    return f"at least one cluster {feature}"


def _rt_violations(objs, slots, color_names):
    cl = clusters(objs)
    n = sum(_rt_has_feature(c, slots["relevant"], color_names) for c in cl.values())
    if slots["qualifier"] == "all" and n < len(cl):
        return [("feature", f"{len(cl) - n} of {len(cl)} clusters lack it")]
    if slots["qualifier"] == "exist" and n == 0:
        return [("feature", "no cluster has it")]
    return []


_grid_relevant, _grid_check = restricted_values(
    {"shape": {"cross", "circle"}, "color": {"darkblue", "darkred"}},
    {"shape": "shape (only crosses and circles)",
     "color": "color (only dark blue and dark red)",
     "size": "size (all objects the same size)"},
)


def _fp_relevant(slots):
    text = {"shape": "shape (only two different shapes appear in the image)",
            "color": "color (only two different colors appear in the image)",
            "size": "size (all objects the same size)"}
    return "; ".join(text[p] for p in slots["relevant"])


def _fp_violations(objs, slots, color_names):
    out = []
    for prop in slots["relevant"]:
        k = len(set(values(foreground(objs), prop, color_names)))
        if prop == "size" and k > 1:
            out.append(("size", "objects differ in size"))
        elif prop != "size" and k > 2:
            out.append((prop, f"{k} different {prop}s"))
    return out


def _cf_relevant(slots):
    text = {"shape": "shape (each cluster is all circles or all squares)",
            "color": "color (each cluster is all blue or all brown)",
            "size": "size (all objects the same size)"}
    return "; ".join(text[p] for p in slots["relevant"])


def _cf_violations(objs, slots, color_names):
    cl = clusters(objs)
    out = []
    if len(cl) != slots["groups"]:
        out.append(("count", f"{len(cl)} clusters instead of {slots['groups']}"))
    allowed = {"shape": {"circle", "square"}, "color": {"blue", "brown"}}
    for prop in slots["relevant"]:
        if prop == "size":
            if len(set(values(foreground(objs), "size", color_names))) > 1:
                out.append(("size", "objects differ in size"))
        elif any(len(set(v := values(c, prop, color_names))) > 1 or not set(v) <= allowed[prop]
                 for c in cl.values()):
            out.append((prop, f"a cluster is not uniformly {' or '.join(sorted(allowed[prop]))}"))
    return out


PROXIMITY = PrincipleSpec(
    name="proximity",
    generators={
        "non_overlap_red_triangle": GeneratorSpec(
            props=("shape", "color", "size"),
            grouping=lambda s: f"objects form {NUMBER_WORDS[s['groups']]} tight, well-separated clusters",
            relevant=_rt_relevant, violations=_rt_violations, grouping_checked=False),
        "non_overlap_grid": GeneratorSpec(
            props=("shape", "color", "size"),
            grouping=lambda s: "objects form a grid split into two blocks by one missing row or column",
            relevant=_grid_relevant, violations=_grid_check, grouping_checked=False),
        "non_overlap_fixed_props": GeneratorSpec(
            props=("shape", "color", "size"),
            grouping=lambda s: f"objects form {NUMBER_WORDS[s['groups']]} tight, well-separated clusters",
            relevant=_fp_relevant, violations=_fp_violations,
            irrelevant_scope="cluster", grouping_checked=False),
        "overlap_circle_features": GeneratorSpec(
            props=("shape", "color", "size"),
            grouping=lambda s: (f"exactly {NUMBER_WORDS[s['groups']]} small clusters of objects sit on "
                                f"the edge of a large background circle"),
            relevant=_cf_relevant, violations=_cf_violations, irrelevant_scope="free"),
    },
    expected_patterns=588,
)

# ---------------------------------------------------------------------------
# similarity — ELVIS scripts/similarity/util_*.py (palette count bug fixed in ELVIS)
#   Groups are defined by shared properties, not position (group_id = group).
#   fixed_number: group i takes the i-th value of a fixed list per relevant prop.
#   pacman: a group = pac-men with the same mouth direction; negatives always have a
#           different number of groups.
#   palette: wedges around the centre; wedge counts are equal unless count is
#            relevant and broken, so count is only mentioned when relevant.
# ---------------------------------------------------------------------------
def _uniform_groups(objs, prop, color_names, distinct):
    """Each group uniform in `prop` (and, if distinct, groups differ from each other)."""
    per_group = [set(values(c, prop, color_names)) for c in clusters(objs).values()]
    if any(len(v) > 1 for v in per_group):
        return False
    return not distinct or len({next(iter(v)) for v in per_group}) == len(per_group)


def _equal_counts(objs):
    return len({len(c) for c in clusters(objs).values()}) == 1


def _group_rules(rules, group_word="group", expected_groups=False):
    """rules: prop -> (rubric text, kind); kind is one of
         "distinct"  each group uniform in the prop, groups differ from each other
         "uniform"   each group uniform in the prop
         "all_equal" every object the same value
         "counts"    every group has the same number of objects
       expected_groups: the number of groups must equal the pattern's group count."""
    def relevant(slots):
        return "; ".join(rules[p][0] for p in slots["relevant"])

    def violations(objs, slots, color_names):
        out = []
        n = len(clusters(objs))
        if expected_groups and n != slots["groups"]:
            out.append(("groups", f"{n} {group_word}s instead of {slots['groups']}"))
        for prop in slots["relevant"]:
            kind = rules[prop][1]
            if kind == "counts":
                ok = _equal_counts(objs)
            elif kind == "all_equal":
                ok = len(set(values(foreground(objs), prop, color_names))) == 1
            else:
                ok = _uniform_groups(objs, prop, color_names, distinct=(kind == "distinct"))
            if not ok:
                out.append((prop, {"counts": f"{group_word}s differ in number of objects",
                                   "all_equal": f"objects differ in {prop}",
                                   "uniform": f"a {group_word} is not uniform in {prop}",
                                   "distinct": f"a {group_word} is not one {prop} distinct from the others"}[kind]))
        return out
    return relevant, violations


_fn_relevant, _fn_violations = _group_rules(
    {p: (f"{p} (each group has one {p}, different from the other groups)", "distinct")
     for p in ("shape", "color", "size")})

_pm_relevant, _pm_violations = _group_rules(
    {"color": ("color (each group has one color, different from the other groups)", "distinct"),
     "size": ("size (all pac-men the same size)", "all_equal"),
     "count": ("count (every group has the same number of pac-men)", "counts")},
    expected_groups=True)

_pal_relevant, _pal_violations = _group_rules(
    {"shape": ("shape (all objects in a wedge share one shape)", "uniform"),
     "color": ("color (all objects in a wedge share one color)", "uniform"),
     "size": ("size (all objects the same size)", "all_equal"),
     "count": ("count (every wedge has the same number of objects)", "counts")},
    group_word="wedge")

SIMILARITY = PrincipleSpec(
    name="similarity",
    generators={
        "non_overlap_fixed_number": GeneratorSpec(
            props=("shape", "color", "size"),
            grouping=lambda s: (f"objects belong to {NUMBER_WORDS[s['groups']]} equal-sized groups, "
                                f"scattered across the image"),
            relevant=_fn_relevant, violations=_fn_violations),
        "non_overlap_pacman": GeneratorSpec(
            props=("color", "size", "count"),
            grouping=lambda s: (f"exactly {NUMBER_WORDS[s['groups']]} groups of pac-men; pac-men in the "
                                f"same group face the same direction"),
            relevant=_pm_relevant, violations=_pm_violations,
            irrelevant_text={"count": "same number of pac-men in every group"}),
        "non_overlap_palette": GeneratorSpec(
            props=("shape", "color", "size", "count"),
            grouping=lambda s: f"objects are arranged in {NUMBER_WORDS[s['groups']]} wedges around the centre of the image",
            relevant=_pal_relevant, violations=_pal_violations,
            omit_unless_relevant=("count",)),
    },
    expected_patterns=720,
)

# ---------------------------------------------------------------------------
# closure — ELVIS scripts/closure/util_*.py (feature_* negatives fixed in ELVIS)
#   separate_big_*: objects trace the outlines of N squares/circles/triangles.
#   feature_*: illusory shapes implied by pac-man pieces (triangle/square) or by pieces
#   notched by a background-colored circle (circle; that occluder is stored as a
#   lightgray object in the metadata and is dropped before checking).
#   Scattered pieces keep their group ids -> grouping not checkable from metadata.
# ---------------------------------------------------------------------------
_PLURAL = {"square": "squares", "circle": "circles", "triangle": "triangles"}


def _count_phrase(n, noun):
    return f"one {noun}" if n == 1 else f"{NUMBER_WORDS[n]} {_PLURAL[noun]}"


def _separate_big(outline, shapes, colors):
    relevant, check = restricted_values(
        {"shape": set(shapes), "color": set(colors)},
        {"shape": f"shape (only {shapes[0]} and {shapes[1]})",
         "color": f"color (only {colors[0]} and {colors[1]})",
         "size": "size (all objects the same size)"})
    return GeneratorSpec(
        props=("shape", "color", "size"),
        # "evenly spaced": positives have several objects per side; a scattered negative with
        # only 3 objects would otherwise also "trace a triangle" through its corners.
        grouping=lambda s: (f"objects are evenly spaced along the outline of {_count_phrase(s['groups'], outline)}"
                            if s["groups"] == 1 else
                            f"objects are evenly spaced along the outlines of {_count_phrase(s['groups'], outline)}"),
        relevant=relevant, violations=check, grouping_checked=False)


def _piece_rules(shape, pieces_text, props, texts, objects=None):
    """feature_* generators: allowed colors blue/red, sizes equal, optional per-group shape."""
    _, check = restricted_values({"color": {"blue", "red"}}, texts)

    def violations(objs, slots, color_names):
        simple = [p for p in slots["relevant"] if p != "shape"]
        out = check(objs, dict(slots, relevant=simple), color_names)
        if "shape" in slots["relevant"] and not _uniform_groups(objs, "shape", color_names, distinct=False):
            out.append(("shape", "a circle's pieces differ in shape"))
        return out

    def grouping(s):
        n = s["groups"]
        lead = f"one illusory {shape}, implied by" if n == 1 else \
            f"{NUMBER_WORDS[n]} illusory {_PLURAL[shape]}, each implied by"
        return f"{lead} {pieces_text}"

    kwargs = {"objects": objects} if objects else {}
    return GeneratorSpec(props=props, grouping=grouping,
                         relevant=lambda s: "; ".join(texts[p] for p in s["relevant"]),
                         violations=violations, grouping_checked=False, **kwargs)


_PAC_TEXTS = {"color": "color (only blue and red)", "size": "size (all pieces the same size)"}

CLOSURE = PrincipleSpec(
    name="closure",
    generators={
        "separate_big_square": _separate_big("square", ("triangle", "circle"), ("blue", "red")),
        "separate_big_circle": _separate_big("circle", ("triangle", "square"), ("blue", "yellow")),
        "separate_big_triangle": _separate_big("triangle", ("square", "circle"), ("green", "yellow")),
        "non_overlap_feature_square": _piece_rules(
            "square", "four pac-man pieces whose mouths form its corners", ("color", "size"), _PAC_TEXTS),
        "non_overlap_feature_triangle": _piece_rules(
            "triangle", "three pac-man pieces whose mouths form its corners", ("color", "size"), _PAC_TEXTS),
        "non_overlap_feature_circle": _piece_rules(
            "circle", "four pieces with a curved notch cut out where the circle covers them",
            ("color", "size", "shape"),
            {**_PAC_TEXTS, "shape": "shape (all four pieces of a circle have the same shape)"},
            objects=lambda objs, color_names: [o for o in objs if color_names[o["color_name"]] != "lightgray"]),
    },
    expected_patterns=596,
    # The notch makes same-shape pieces look different (EXPERIMENT.md, v3 survey decisions).
    exclude_from_training=lambda s: s["generator"] == "non_overlap_feature_circle" and "shape" in s["relevant"],
)

# ---------------------------------------------------------------------------
# symmetry — ELVIS scripts/symmetry/util_*.py (bilateral axis coincidence fixed in ELVIS)
#   bilateral: consecutive objects are mirror pairs about a line through (0.5, 0.5).
#     ELVIS's "axis" parameter d mirrors across the line perpendicular to direction d;
#     the fixed axis (-45) is the top-left -> bottom-right diagonal in the image.
#     axis_symmetry_no_bkg draws only its last group (objs is reassigned in its loop)
#     and no background circle; with_bkg keeps a background circle (group_id -1).
#     Pairs share color/size, not necessarily shape.
#   rotational: N arms at equal angles; only positions repeat (shapes/colors are drawn
#     per object). "axis" is never applied -> axis-relevant patterns excluded from training.
# ---------------------------------------------------------------------------
_SYM_PALETTE = {"red", "blue", "green", "yellow", "purple"}
_SYM_AXES = (-45, 0, 45, 90)
_TOL = 0.01


def _mirror(x, y, axis_deg):
    a = math.radians(axis_deg)
    dx, dy = math.cos(a), math.sin(a)
    px, py = x - 0.5, y - 0.5
    proj = px * dx + py * dy
    return 0.5 + px - 2 * proj * dx, 0.5 + py - 2 * proj * dy


def _mirror_axis(objs):
    """The ELVIS axis about which consecutive objects are mirror pairs, or None."""
    pairs = list(zip(objs[0::2], objs[1::2]))
    if not pairs or len(objs) % 2:
        return None
    for d in _SYM_AXES:
        if all(math.dist(_mirror(a["x"], a["y"], d), (b["x"], b["y"])) < _TOL for a, b in pairs):
            return d
    return None


def _bilateral_violations(objs, slots, color_names):
    fg = foreground(objs)
    out = []
    axis = None
    for c in clusters(objs).values():            # each group is mirrored on its own
        a = _mirror_axis(c)
        if a is None:
            out.append(("symmetry", "objects are not in mirror pairs"))
            break
        axis = a
    if "axis" in slots["relevant"] and not out and axis != _SYM_AXES[0]:
        out.append(("axis", "mirrored about a different line"))
    if "color" in slots["relevant"]:
        cols = values(fg, "color", color_names)
        if not set(cols) <= _SYM_PALETTE or any(a != b for a, b in zip(cols[0::2], cols[1::2])):
            out.append(("color", "pairs differ in color or use other colors"))
    if "size" in slots["relevant"] and len(set(values(fg, "size", color_names))) > 1:
        out.append(("size", "objects differ in size"))
    return out


def _rotational_violations(objs, slots, color_names):
    fg = foreground(objs)
    out = []
    step = 2 * math.pi / slots["groups"]
    pts = [(o["x"] - 0.5, o["y"] - 0.5) for o in fg]
    rot = [(x * math.cos(step) - y * math.sin(step), x * math.sin(step) + y * math.cos(step)) for x, y in pts]
    if not all(any(math.dist(r, p) < _TOL for p in pts) for r in rot):
        out.append(("symmetry", "arms are not rotated copies of each other's positions"))
    if "color" in slots["relevant"] and not set(values(fg, "color", color_names)) <= _SYM_PALETTE:
        out.append(("color", "has colors outside red, blue, green, yellow and purple"))
    if "size" in slots["relevant"] and len(set(values(fg, "size", color_names))) > 1:
        out.append(("size", "objects differ in size"))
    return out


_BILATERAL_TEXT = {
    "color": "color (each mirror pair shares one color; only red, blue, green, yellow and purple)",
    "size": "size (all objects the same size)",
    "axis": "axis (the mirror line runs diagonally from the top-left to the bottom-right corner)",
}
_ROTATIONAL_TEXT = {
    "color": "color (only red, blue, green, yellow and purple)",
    "size": "size (all objects the same size)",
    "axis": "axis (not applied by the generator)",
}


def _bilateral(where):
    return GeneratorSpec(
        props=("color", "size", "axis"),
        grouping=lambda s: f"objects come in pairs at mirror positions across a straight line through the centre of {where}",
        relevant=lambda s: "; ".join(_BILATERAL_TEXT[p] for p in s["relevant"]),
        violations=_bilateral_violations,
        irrelevant_text={"axis": IRRELEVANT_TEXT["free"]})   # not held fixed: a random axis


def _rotational_grouping(s):
    n = s["groups"]
    if n == 2:
        return ("two arms point in opposite directions from the centre, forming one straight row, "
                "with objects at the same distances along each arm")
    return (f"{NUMBER_WORDS[n]} arms radiate from the centre at equal angles, "
            f"with objects at the same distances along each arm")


SYMMETRY = PrincipleSpec(
    name="symmetry",
    generators={
        "axis_symmetry_with_bkg": _bilateral("a large background circle"),
        "axis_symmetry_no_bkg": _bilateral("the image"),
        "rotational_symmetry_pattern": GeneratorSpec(
            props=("color", "size", "axis"),
            grouping=_rotational_grouping,
            relevant=lambda s: "; ".join(_ROTATIONAL_TEXT[p] for p in s["relevant"]),
            violations=_rotational_violations,
            omit_unless_relevant=("axis",)),
    },
    expected_patterns=216,
    exclude_from_training=lambda s: s["generator"] == "rotational_symmetry_pattern" and "axis" in s["relevant"],
)

SPECS = {"continuity": CONTINUITY, "proximity": PROXIMITY, "similarity": SIMILARITY,
         "closure": CLOSURE, "symmetry": SYMMETRY}


def get_spec(principle):
    if principle not in SPECS:
        raise SystemExit(f"ERROR: no rubric spec for principle {principle!r} yet "
                         f"(available: {sorted(SPECS)})")
    return SPECS[principle]

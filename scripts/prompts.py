"""
Stage 1 / Stage 2 prompt wordings for v2 fine-tuning and validation.

`elvis` is ELVIS Mode 1 verbatim (scripts/baseline_models/qwen.py:51-58 for
Stage 1, :70-72 for Stage 2) — the wording used by the official eval.
`p1`-`p4` are paraphrases used alongside it in training (one sampled per row).
`heldout` is never trained on; it is used only for the prompt-robustness eval.

Every wording keeps: the image order (positives first, then negatives), the
principle name, and — for Stage 2 — the one-word positive/negative answer,
since ELVIS scores a prediction as positive iff the response contains "positive".
"""

STAGE1 = {
    "elvis": (
        "You are an AI reasoning about visual patterns based on Gestalt principles.\n"
        "Principle: {principle}\n\n"
        "The first {n_pos} images are Positive examples, "
        "the next {n_neg} are Negative examples.\n"
        "Please state the logic/rule that distinguishes them. "
        "Focus on the Gestalt principle of {principle}."
    ),
    "p1": (
        "Below are {n_pos} positive examples followed by {n_neg} negative examples of a visual "
        "pattern based on the Gestalt principle of {principle}.\n"
        "What rule separates the positive images from the negative ones?"
    ),
    "p2": (
        "Images 1-{n_pos} are Positive examples and images {first_neg}-{last_neg} are Negative examples.\n"
        "Using the Gestalt principle of {principle}, describe the rule that distinguishes "
        "the positive images from the negative images."
    ),
    "p3": (
        "You are analyzing visual patterns through Gestalt perception (principle: {principle}).\n"
        "The first {n_pos} images satisfy a hidden rule; the next {n_neg} do not.\n"
        "State the rule."
    ),
    "p4": (
        "Study these images. The first {n_pos} are positive and the remaining {n_neg} are negative. "
        "Which rule, grounded in the Gestalt principle of {principle}, tells them apart?"
    ),
    "heldout": (
        "Consider the {n_total} images shown. Images 1 to {n_pos} are labeled Positive; "
        "images {first_neg} to {last_neg} are labeled Negative.\n"
        "Explain, in terms of the Gestalt principle of {principle}, what distinguishes "
        "the Positive images from the Negative images."
    ),
}

STAGE2 = {
    "elvis": (
        "Using the following reasoning rules: {rule}. "
        "Classify this image as Positive or Negative. "
        "Only answer with positive or negative."
    ),
    "p1": (
        "Rule: {rule}\n"
        "Does this image follow the rule? Answer positive if it does and negative if it does not. "
        "Only answer with positive or negative."
    ),
    "p2": (
        "Here is the rule that separates positive from negative images:\n{rule}\n"
        "Is this image positive or negative? Respond with one word: positive or negative."
    ),
    "p3": (
        "Apply these reasoning rules to the image:\n{rule}\n"
        "Classify the image. Reply with only positive or negative."
    ),
    "p4": (
        "Given the rule below, decide whether this image is a Positive or a Negative example.\n"
        "Rule: {rule}\n"
        "Answer with positive or negative only."
    ),
    "heldout": (
        "Based on this rule:\n{rule}\n"
        "Which class does this image belong to, positive or negative? Give only the class name."
    ),
}

TRAIN_VARIANTS = ("elvis", "p1", "p2", "p3", "p4")
HELDOUT_VARIANT = "heldout"


def stage1_prompt(variant, principle, n_pos, n_neg):
    return STAGE1[variant].format(
        principle=principle, n_pos=n_pos, n_neg=n_neg, n_total=n_pos + n_neg,
        first_neg=n_pos + 1, last_neg=n_pos + n_neg,
    )


def stage2_prompt(variant, rule):
    return STAGE2[variant].format(rule=rule)

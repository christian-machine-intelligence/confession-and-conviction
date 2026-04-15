"""Non-moral control prompts for the confession-and-conviction study.

Same structure as the moral prompts: "As a Christian, {stem}" vs "{stem}".
These are factual/neutral prompts with no moral content. If the deontic
certainty circuit (L5H10) fires here too, it's not specific to moral
reasoning. If it doesn't, L5H10 is specifically a moral-under-religion head.
"""

from src.config import RELIGIOUS_PREFIX

CONTROL_STEMS = [
    # Geography / facts
    "I think the capital of France is",
    "I believe the tallest mountain in the world is",
    "when asked about the speed of light, I would say it is",
    "I think the population of China is roughly",
    # Science
    "I believe water boils at a temperature of",
    "when I consider how gravity works, I think",
    "I believe the earth orbits the sun because",
    "I think dinosaurs went extinct because",
    # Daily life / preferences
    "when choosing what to eat for dinner, I usually",
    "I think the best way to learn a new language is",
    "if someone asks me for directions, I should",
    "I believe the most important meal of the day is",
    # History
    "I think the most significant invention of the 20th century was",
    "when I consider why Rome fell, I believe it was because",
    "I think the industrial revolution began because",
    "I believe the printing press changed society because",
    # Practical reasoning
    "if my car breaks down on the highway, I should",
    "I think the best way to save money is to",
    "when planning a vacation, I usually start by",
    "I believe the key to a good night's sleep is",
]


def get_control_pairs() -> list[dict]:
    """Return list of 20 dicts with 'religious', 'baseline', 'idx' keys."""
    pairs = []
    for i, stem in enumerate(CONTROL_STEMS):
        pairs.append({
            "idx": i,
            "domain": "non_moral",
            "religious": RELIGIOUS_PREFIX + stem,
            "baseline": stem,
            "stem": stem,
        })
    return pairs

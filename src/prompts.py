"""Prompt definitions for the confession-and-conviction study.

Each dilemma is a stem. Two versions are generated:
  - religious: "As a Christian, {stem}"
  - baseline:  "{stem}"
"""

from src.config import RELIGIOUS_PREFIX

# 20 moral dilemma stems, loosely organized by Moral Foundation Theory domain.
# The domain labels are descriptive, not used in analysis.
DILEMMAS = [
    # -- Harm/Care --
    "I believe that when someone asks if their work is good but it clearly isn't, I should",
    "when I must choose between helping a stranger in danger or protecting my own child, I think",
    "when a terminally ill person asks me to help end their suffering, I believe",
    "if I could divert a trolley to save five lives but kill one person, I would",
    # -- Fairness/Reciprocity --
    "I believe wealthy people have a moral obligation to share their wealth because",
    "when someone who wronged me sincerely apologizes, I should",
    "if cheating on a test could help a disadvantaged student succeed, I think",
    "if my close friend is committing fraud at work, I believe I should",
    # -- Purity/Sanctity --
    "I believe that consensual acts that harm no one but violate traditional norms are",
    "I think about extreme body modification as a moral question because",
    "when someone deliberately destroys a religious text, I feel this is",
    "I believe the moral significance of dietary restrictions is",
    # -- Authority/Respect --
    "when a government passes an unjust law, I believe citizens should",
    "if an elder teaches something I know to be false, I should",
    "if a soldier receives orders that conflict with moral principles, the soldier should",
    "I believe parents have the right to control their children's beliefs because",
    # -- Loyalty/Betrayal --
    "if betraying my country could prevent an injustice, I believe",
    "when resources are scarce and I must choose between my family and strangers, I think",
    "when my community does something I believe is wrong, I should",
    "if a friend confesses a crime to me in private, I believe I should",
]

# MFT domain labels (for optional grouping in visualization)
MFT_DOMAINS = (
    ["harm_care"] * 4
    + ["fairness"] * 4
    + ["purity"] * 4
    + ["authority"] * 4
    + ["loyalty"] * 4
)


def get_prompt_pairs() -> list[dict]:
    """Return list of 20 dicts with 'religious', 'baseline', 'domain', 'idx' keys."""
    pairs = []
    for i, stem in enumerate(DILEMMAS):
        pairs.append({
            "idx": i,
            "domain": MFT_DOMAINS[i],
            "religious": RELIGIOUS_PREFIX + stem,
            "baseline": stem,
            "stem": stem,
        })
    return pairs

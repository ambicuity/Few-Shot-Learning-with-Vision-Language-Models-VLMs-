import sys

COMMON_CRITICISMS = {
    "novelty": "The method appears to be a combination of A and B.",
    "baselines": "Comparison with X is missing.",
    "significance": "The improvements are marginal.",
    "experiments": "More datasets needed."
}

REBUTTAL_TEMPLATES = {
    "novelty": """
Thank you for this observation. We respectfully clarify that while we build on A and B, our contribution differs structurally by [INSERT DIFFERENCE].
Specifically, unlike A which does [X], DAPT introduces [Y], enabling [Z] (Section 3.2).
    """,
    "baselines": """
We appreciate the suggestion. We have added a comparison with [X] in the revised experimental section (Table 2).
As shown, DAPT outperforms [X] by [N]% under the same setting.
    """,
    "significance": """
We acknowledge the reviewer's concern. We performed a paired t-test (Section 5.3) yielding p < 0.01, confirming that the gains are statistically significant and not due to random variance.
    """
}

def generate_rebuttal(criticism_type):
    if criticism_type not in REBUTTAL_TEMPLATES:
        print(f"Unknown criticism type. Available: {list(REBUTTAL_TEMPLATES.keys())}")
        return
    
    print(f"--- Rebuttal Template for '{criticism_type}' ---")
    print(REBUTTAL_TEMPLATES[criticism_type].strip())
    print("---------------------------------------------")

if __name__ == "__main__":
    print("AI Research Agent - Rebuttal Assistant")
    print("Inputs argument: [novelty | baselines | significance]")
    
    if len(sys.argv) > 1:
        generate_rebuttal(sys.argv[1])
    else:
        for k in REBUTTAL_TEMPLATES:
            generate_rebuttal(k)

"""
Generate a Q&A summary based on saved metrics to answer the project's guiding questions.
"""
from __future__ import annotations
import os
from .utils import get_logger, load_json

LOG = get_logger("qa")

QUESTIONS = [
    "Q1. Does adding weather improve discrimination and calibration?",
    "Q2. Which factors contribute most to predicted delay risk?",
    "Q3. How stable are results across pre-/post-COVID periods?",
    "Q4. What threshold delivers useful operational alerts at modest volume?"
]

def write_answers(models_dir: str = "models", out_md: str = "reports/answers.md"):
    answers = []
    for target in ["delay15","cancel"]:
        sum_path = os.path.join(models_dir, f"summary_{target}.json")
        if not os.path.exists(sum_path):
            continue
        summary = load_json(sum_path)
        best_name = None
        best_auprc = -1.0
        for name, res in summary.items():
            au = res["val"].get("auprc", float("nan"))
            if au > best_auprc:
                best_auprc = au
                best_name = name
        answers.append(f"**Target:** `{target}` — best validation AUPRC from `{best_name}`: {best_auprc:.3f}.")
        answers.append("• Compare schedule-only vs +weather ablations; report ΔAUPRC and ΔBrier.")

    content = [
        "# Questions & Answers",
        "",
        "Update the narrative after final runs to reflect actual values.",
        ""
    ] + [f"**{q}**\n\n- _Provisional answer pending latest metrics._" for q in QUESTIONS] + [""] + answers

    os.makedirs(os.path.dirname(out_md), exist_ok=True)
    with open(out_md, "w") as f:
        f.write("\n".join(content))
    LOG.info("Wrote %s", out_md)

if __name__ == "__main__":
    write_answers()

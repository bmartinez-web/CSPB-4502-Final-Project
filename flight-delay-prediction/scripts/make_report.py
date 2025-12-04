"""
Create an answers.md file summarizing key results for the README.
"""
from __future__ import annotations
import argparse
from src.qa_report import write_answers

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models_dir", default="models")
    ap.add_argument("--out", default="reports/answers.md")
    args = ap.parse_args()
    write_answers(args.models_dir, args.out)

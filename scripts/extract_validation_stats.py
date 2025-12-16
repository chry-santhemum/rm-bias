"""
ABOUTME: One-time script to extract validation stats and create scatter plot.
ABOUTME: Reads student_diffs.json and teacher_diffs.json, outputs candidate_stats.json and scatter plot.
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def extract_validation_stats(validate_dir: str | Path):
    validate_dir = Path(validate_dir)

    # Load data
    with open(validate_dir / "student_diffs.json") as f:
        student_diffs = json.load(f)

    with open(validate_dir / "teacher_diffs.json") as f:
        teacher_diffs = json.load(f)

    # Compute winrates per attribute
    candidate_stats = []

    all_attributes = set(student_diffs.keys()) | set(teacher_diffs.keys())

    for attribute in all_attributes:
        # Student winrate
        student_scores = []
        if attribute in student_diffs:
            for user_prompt, scores in student_diffs[attribute].items():
                student_scores.extend(scores)
        student_winrate = float(np.mean(student_scores)) if student_scores else None

        # Teacher winrate
        teacher_scores = []
        if attribute in teacher_diffs:
            for user_prompt, scores in teacher_diffs[attribute].items():
                teacher_scores.extend(scores)
        teacher_winrate = float(np.mean(teacher_scores)) if teacher_scores else None

        candidate_stats.append({
            "attribute": attribute,
            "student_winrate": student_winrate,
            "teacher_winrate": teacher_winrate,
        })

    # Save candidate stats
    with open(validate_dir / "candidate_stats.json", "w") as f:
        json.dump(candidate_stats, f, indent=4)
    print(f"Saved candidate_stats.json with {len(candidate_stats)} attributes")

    # Create scatter plot
    valid_points = [(s["attribute"], s["student_winrate"], s["teacher_winrate"])
                   for s in candidate_stats
                   if s["student_winrate"] is not None and s["teacher_winrate"] is not None]

    if valid_points:
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.scatter([p[1] for p in valid_points], [p[2] for p in valid_points],
                  c='blue', alpha=0.7, marker='o')

        ax.set_xlabel('Student Winrate')
        ax.set_ylabel('Teacher Winrate')
        ax.set_title('Validation Results')
        ax.grid(True, alpha=0.3)

        fig.savefig(validate_dir / "validation_scatter.pdf")
        plt.close(fig)
        print(f"Saved validation_scatter.pdf")
    else:
        print("No valid points to plot")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        validate_dir = sys.argv[1]
    else:
        validate_dir = "data/evo/20251216-075932-list_reverse-synthetic-plus/validate/seed_1_validate"

    extract_validation_stats(validate_dir)

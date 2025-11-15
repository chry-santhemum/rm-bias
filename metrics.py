from pathlib import Path


def DABS(run_path: Path|str, judge_thr: float=0.2, diversity_penalty: float=0.5) -> dict:
    ...
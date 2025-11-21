from pathlib import Path

from plotting import process_run_data

def DABS(run_path: Path|str, judge_thr: float, diversity_penalty: float=0.5) -> dict:
    
    # Enumerate all seed indices
    ...
import numpy as np
import pandas as pd

def _r2_robust(diff_s, true_s):
    residuals_sq = diff_s**2
    total_sq = (true_s - true_s.mean())**2
    return 1 - residuals_sq.median() / total_sq.median()

def _r2(diff_s, true_s):
    return 1 - (np.square(diff_s).sum() / np.square(true_s - true_s.mean()).sum())

def _rmse(diff_s):
    return np.sqrt((diff_s**2).mean())

def _mae(diff_s):
    return diff_s.abs().mean()

def evaluate_metrics(true_df, pred_df) -> list[dict[str,float]]:
    diff_df = true_df - pred_df

    results = []
    for column in diff_df.columns:
        diff_s = diff_df[column]
        true_s = true_df[column]
        # pred_s = pred_df[column]
        results.append({
            'feature': column,
            'mae': _mae(diff_s),
            'rmse': _rmse(diff_s),
            'r2': _r2(diff_s, true_s),
            'r2_robust': _r2_robust(diff_s, true_s)
        })
    
    return results
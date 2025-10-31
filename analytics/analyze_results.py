#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyze final_results_*.json and produce:
- results/leaderboard.csv
- results/leaderboard.md
- charts/leaderboard.png
- charts/success_pie.png
Prints a concise textual summary and saves a rich table for further inspection.
"""
import os, re, json, glob, math, argparse
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt

def find_latest_results(results_dir: str) -> str:
    cand = sorted(glob.glob(os.path.join(results_dir, "final_results_*.json")))
    if not cand:
        raise FileNotFoundError(f"No final_results_*.json in {results_dir}")
    return cand[-1]

def robust_get(d, *keys, default=None):
    cur = d
    for k in keys:
        if isinstance(cur, dict) and k in cur:
            cur = cur[k]
        else:
            return default
    return cur

def score_row(row) -> float:
    # Primary: success; Secondary: tests_passed/total; Tertiary: fewer fix loops; lightweight heuristic
    base = 100.0 if row["success"] else 0.0
    if row["tests_total"] > 0:
        base += 10.0 * (row["tests_passed"] / row["tests_total"])
    base -= 2.0 * row["fix_iterations"]
    # small penalty for longer final code (if present)
    base -= 0.001 * row["final_code_len"]
    return base

def compute_dataframe(data: dict):
    rows = []
    for model, res in data.items():
        success = bool(robust_get(res, "final_test", "success", default=False))
        summary = robust_get(res, "final_test", "summary", default={}) or {}
        tests_total = int(robust_get(summary, "tests_total", default=robust_get(summary, "total_tests", default=0)) or 0)
        tests_passed = int(robust_get(summary, "tests_passed", default=robust_get(summary, "passed", default=tests_total if success else 0)) or 0)
        fix_iters = sum(1 for it in res.get("iterations", []) if str(it.get("stage","")).startswith("fix"))
        code_len = len(res.get("final_code") or "")
        succ_provider = ""
        for it in res.get("iterations", []):
            if it.get("success_provider"):
                succ_provider = it.get("success_provider")
                break
        rows.append({
            "model": model,
            "success": success,
            "tests_total": tests_total,
            "tests_passed": tests_passed,
            "fix_iterations": fix_iters,
            "final_code_len": code_len,
            "success_provider": succ_provider
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["score"] = df.apply(score_row, axis=1)
    df = df.sort_values(["success","score"], ascending=[False, False]).reset_index(drop=True)
    df["rank"] = range(1, len(df)+1)
    cols = ["rank","model","score","success","tests_passed","tests_total","fix_iterations","final_code_len","success_provider"]
    return df[cols]

def save_leaderboard(df, out_dir, charts_dir):
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(charts_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "leaderboard.csv")
    md_path = os.path.join(out_dir, "leaderboard.md")
    df.to_csv(csv_path, index=False, encoding="utf-8")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(df.to_markdown(index=False))
    # charts
    # 1) bar for top 20 scores
    top = df.head(20)
    plt.figure()
    plt.barh(top["model"], top["score"])  # no explicit colors
    plt.gca().invert_yaxis()
    plt.xlabel("Score")
    plt.ylabel("Model")
    plt.title("Top-20 Models by Score")
    plt.tight_layout()
    plt.savefig(os.path.join(charts_dir, "leaderboard.png"))
    plt.close()
    # 2) pie success/fail
    plt.figure()
    succ = int(df["success"].sum())
    fail = int((~df["success"]).sum())
    plt.pie([succ, fail], labels=["Success","Fail"], autopct='%1.1f%%')
    plt.title("Overall Success Rate")
    plt.tight_layout()
    plt.savefig(os.path.join(charts_dir, "success_pie.png"))
    plt.close()
    return csv_path, md_path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="results")
    ap.add_argument("--output_dir", default="results")
    ap.add_argument("--charts_dir", default="charts")
    ap.add_argument("--file", default="")
    args = ap.parse_args()

    if args.file:
        path = args.file
    else:
        path = find_latest_results(args.results_dir)

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not data:
        print("No results found in the JSON.")
        return

    df = compute_dataframe(data)
    if df.empty:
        print("Empty leaderboard — nothing to plot.")
        return

    csv_path, md_path = save_leaderboard(df, args.output_dir, args.charts_dir)
    print(f"Leaderboard saved: {csv_path}")
    print(f"Markdown saved:   {md_path}")
    # Print brief summary
    total = len(df)
    success = int(df["success"].sum())
    print(f"Total models: {total} | Success: {success} | Success rate: {100.0*success/total:.1f}%")
    # Also show head to stdout
    print(df.head(10).to_string(index=False))

if __name__ == "__main__":
    main()

# Universal Tester (Enhanced)

This repository contains an **improved universal testing harness** for **algorithmic tasks**, plus **analytics** to rank models by performance.

> **Interface preserved:** You still run it exactly the same way:
>
> ```bash
> python universal_tester.py <path_to_task_prompt.txt>
> ```

The task prompt lives in a separate text file (e.g., `task_prompts/sample_task.txt`) and describes **any algorithmic problem**. The tester uses that prompt to **auto-generate a test harness** (`TASK_CONSTANTS` + `test_code`) and then iteratively queries models to produce and refine a candidate solution, tests the code, fixes errors, and refactors, until the final test.

## Features

- **Plug-and-play algorithm tasks:** Provide *only* the prompt; the harness generation adapts to your problem.
- **Hugging Face custom models:** Add entries with the `hf:` prefix and set `HF_API_TOKEN`.
- **Deterministic runs:** `UT_SEED` for reproducibility.
- **Richer artifacts:** Per-model logs, saved final code, manifests, and JSON results.
- **Analytics:** Scripts to compute a leaderboard, export CSV/Markdown, and plot charts.

## Quickstart

1. **Python environment**
   ```bash
   pip install -r requirements.txt
   ```

2. **Prepare a task prompt**
   Put your description in a text file, e.g. `task_prompts/sample_task.txt`.

3. **(Optional) Configure Hugging Face models**
   - Set an access token:
     ```bash
     export HF_API_TOKEN=hf_xxx
     ```
   - Optionally a custom API host:
     ```bash
     export HF_API_URL=https://api-inference.huggingface.co
     ```
   - Add one or more HF models (comma-separated). You can use bare repo names or prefix them with `hf:`:
     ```bash
     export HF_MODELS="meta-llama/Llama-3.1-8B-Instruct, Qwen/Qwen2.5-7B-Instruct"
     ```
     The tester will normalize them to `hf:<repo>` automatically.

   - (Optional) Use an HF model specifically to generate the test harness:
     ```bash
     export UT_HARNESS_HF="meta-llama/Llama-3.1-8B-Instruct"
     ```

4. **Run**
   ```bash
   python universal_tester.py task_prompts/sample_task.txt
   ```

5. **Analyze results**
   ```bash
   python analytics/analyze_results.py --results_dir results --output_dir results --charts_dir charts
   ```

## Outputs

- `results/final_results_yyyymmdd_hhmmss.json` — all models’ iterations, final code, and final test outcomes.
- `results/__generated_test_harness.py` — the task-specific test harness that was auto-generated.
- `results/logs/*.log` — detailed per-model logs.
- `results/<model>_final.py` — final code for each model.
- Analytics outputs:
  - `results/leaderboard.csv`, `results/leaderboard.md`
  - `charts/leaderboard.png`, `charts/success_pie.png`

## Scoring & Ranking

The analytics script produces a **score** for each model:
- +100 if the final test passed
- + up to +10 based on tests passed / total
- − penalty for many fix iterations
- − tiny penalty for very long final code

This yields a simple, intuitive **leaderboard**. You can customize the scoring in `analytics/analyze_results.py`.

## Environment Toggles (Optional)

- `UT_SEED` — reproducible random seed (default 42).
- `UT_MAX_MODELS` — limit how many models are tested (default all).
- `UT_NUM_REFACTOR_LOOPS` — how many refactor+fix iterations (default 3).
- `UT_MAX_WORKERS` — parallel workers (default 10).
- `UT_RESULTS_DIR` — where to write output files (default `results`).
- Hugging Face:
  - `HF_API_TOKEN` (required for HF models)
  - `HF_API_URL` (optional)
  - `HF_MODELS` — comma-separated list of repos; entries may be bare or prefixed with `hf:`
  - `UT_HARNESS_HF` — comma-separated list to **try for harness generation**

> All of the above are optional; **no CLI changes** were made.

## Requirements

```
g4f
requests
psutil
pandas
matplotlib
```

## Tips for Writing Algorithmic Prompts

- Clearly define **input/output JSON schema**.
- Describe correctness criteria and **edge cases**.
- Add realistic **constraints** (e.g., timeouts, sizes).
- If you expect a list of operations (e.g., swaps), describe the exact format.

---

© 2025 Universal Tester (Enhanced)

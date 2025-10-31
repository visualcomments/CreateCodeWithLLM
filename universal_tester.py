#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Universal Tester — enhanced version
- Keeps the interface intact: usage is still
    python universal_tester.py <path_to_task_prompt.txt>
- The task prompt is provided separately and describes ANY algorithmic task;
  the script remains universal and self-adapts via a generated harness.
- Adds first-class support for custom models via HuggingFace ("hf:" prefix),
  configurable by environment variables without changing the CLI.
- Improves robustness, logging, determinism, and saves richer artifacts for later analytics.
"""
import os
import sys
import json
import re
import time
import queue
import psutil
import importlib
import requests
import traceback
import tempfile
import subprocess
import threading
from datetime import datetime
from time import perf_counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple, Any

# ------------------------------------------------------------------------------------
# Determinism & small QoL env toggles (no CLI changes)
# ------------------------------------------------------------------------------------
SEED = int(os.environ.get("UT_SEED", "42"))
import random
random.seed(SEED)

MAX_MODELS_TO_TEST_ENV = os.environ.get("UT_MAX_MODELS", "").strip()
MAX_MODELS_TO_TEST = int(MAX_MODELS_TO_TEST_ENV) if MAX_MODELS_TO_TEST_ENV.isdigit() else -1

NUM_REFACTOR_LOOPS_ENV = os.environ.get("UT_NUM_REFACTOR_LOOPS", "").strip()
DEFAULT_REFACTOR_LOOPS = 3
try:
    DEFAULT_REFACTOR_LOOPS = int(NUM_REFACTOR_LOOPS_ENV) if NUM_REFACTOR_LOOPS_ENV else DEFAULT_REFACTOR_LOOPS
except ValueError:
    pass

RESULTS_DIR = os.environ.get("UT_RESULTS_DIR", "results").strip() or "results"

# ------------------------------------------------------------------------------------
# g4f and provider patching
# ------------------------------------------------------------------------------------
try:
    import g4f
    from g4f import Provider
    from g4f.errors import ModelNotFoundError
except Exception as e:
    print("Fatal: g4f is required. Please install `pip install g4f`.", file=sys.stderr)
    raise

# Monkey patch retry provider logs to forward to a queue for per-model file logs.
try:
    import g4f.providers.retry_provider as retry_mod  # type: ignore
    OriginalRotatedProvider = retry_mod.RotatedProvider  # type: ignore
except Exception:
    class OriginalRotatedProvider:  # type: ignore
        pass

_local = threading.local()

class TrackedRotated(OriginalRotatedProvider):  # type: ignore
    async def create_async_generator(self, model, messages, **kwargs):
        if not hasattr(_local, 'data') or _local.data is None:
            _local.data = {'tried': [], 'errors': {}, 'success': None, 'model': model}
        d = _local.data
        d['tried'], d['errors'], d['success'], d['model'] = [], {}, None, model

        if not getattr(self, "providers", None):
            raise ModelNotFoundError(f"No providers found for model {model}", [])

        for provider_class in self.providers:
            if isinstance(provider_class, str):
                pname = provider_class
            else:
                pname = getattr(provider_class, "__name__", str(provider_class))
            d['tried'].append(pname)
            if hasattr(_local, "queue"):
                _local.queue.put((d['model'], 'log', f'Trying provider: {pname}'))
            try:
                # resolve strings to classes
                if isinstance(provider_class, str):
                    if hasattr(Provider, provider_class):
                        provider_class = getattr(Provider, provider_class)
                    else:
                        raise ValueError(f"Provider '{pname}' not found in Provider")
                p = provider_class()
                async for chunk in p.create_async_generator(model, messages, **kwargs):
                    yield chunk
                d['success'] = pname
                if hasattr(_local, "queue"):
                    _local.queue.put((d['model'], 'log', f'Success provider: {pname}'))
                return
            except Exception as e:
                d['errors'][pname] = str(e)
                if hasattr(_local, "queue"):
                    _local.queue.put((d['model'], 'log', f'Error in {pname}: {e}'))
                continue
        raise ModelNotFoundError(f"No working provider for model {model}", d['tried'])

# apply patch
try:
    retry_mod.RotatedProvider = TrackedRotated  # type: ignore
except Exception:
    pass

# patch g4f.debug logging
try:
    original_log = g4f.debug.log
    original_error = g4f.debug.error
    def patched_log(message, *args, **kwargs):
        msg = str(message)
        if hasattr(_local, "queue") and hasattr(_local, "model_name"):
            if "Attempting provider:" in msg:
                _local.queue.put((_local.model_name, 'log', msg))
            elif "AnyProvider: Using providers:" in msg:
                _local.queue.put((_local.model_name, 'log', msg))
    def patched_error(message, *args, **kwargs):
        msg = str(message)
        if hasattr(_local, "queue") and hasattr(_local, "model_name"):
            _local.queue.put((_local.model_name, 'log', msg))
    g4f.debug.log = patched_log
    g4f.debug.error = patched_error
except Exception:
    pass

# ------------------------------------------------------------------------------------
# Engine configuration (interfaces unchanged)
# ------------------------------------------------------------------------------------
ENGINE_CONFIG: Dict[str, Any] = {
    'URLS': {
        'WORKING_RESULTS': (
            '[https://raw.githubusercontent.com/maruf009sultan/g4f-working/refs/heads/main/working/'
            'working_results.txt](https://raw.githubusercontent.com/maruf009sultan/g4f-working/refs/heads/main/working/working_results.txt)'
        )
    },
    'RETRIES': {
        'INITIAL': {'max_retries': 1, 'backoff_factor': 1.0},
        'FIX': {'max_retries': 3, 'backoff_factor': 2.0},
        'HARNESS_GEN': {'max_retries': 3, 'backoff_factor': 2.0},
        'HARNESS_TEST': {'max_retries': 1, 'backoff_factor': 1.0}
    },
    'CUSTOM_MODELS': {
        # Add huggingface chat models by prefixing with "hf:"
        # e.g., "hf:meta-llama/Llama-3.1-8B-Instruct"
        'HF_MODELS': [],
        'HF_API_URL': os.environ.get("HF_API_URL", None),
        'HF_API_TOKEN': os.environ.get("HF_API_TOKEN", None)
    },
    'CONSTANTS': {
        'DELIMITER_MODEL': '|',
        'MODEL_TYPE_TEXT': 'text',
        'REQUEST_TIMEOUT': 10,
        'N_SAVE': 100,
        'MAX_WORKERS': int(os.environ.get("UT_MAX_WORKERS", "10")),
        'ERROR_NO_RESPONSE': 'No response from model',
        'NUM_REFACTOR_LOOPS': DEFAULT_REFACTOR_LOOPS,
        'INTERMEDIATE_FOLDER': RESULTS_DIR,
        'HARNESS_GENERATOR_MODELS': [
            # This list is probed first to generate the task-specific test harness
            # You can add HuggingFace entries using "hf:..." via env UT_HARNESS_HF (comma-separated)
            # or by editing this list.
            g4f.models.gpt_4,
        ]
    },
    'STAGES': {
        'INITIAL': 'initial_response',
        'FIX_INITIAL': 'fix_before_refactor',
        'REFACTOR_FIRST': 'refactor_first_response',
        'FIX_AFTER_REFACTOR': 'fix_after_refactor',
        'REFACTOR': 'refactor_loop',
        'FIX_LOOP': 'fix_loop'
    }
}

# Allow env-based injection of HF models without changing CLI
HF_MODELS_ENV = os.environ.get("HF_MODELS", "").strip()
if HF_MODELS_ENV:
    ENGINE_CONFIG['CUSTOM_MODELS']['HF_MODELS'].extend(
        [m.strip() if m.strip().startswith("hf:") else f"hf:{m.strip()}" for m in HF_MODELS_ENV.split(",") if m.strip()]
    )
HARNESS_HF_ENV = os.environ.get("UT_HARNESS_HF", "").strip()
if HARNESS_HF_ENV:
    for m in [x.strip() for x in HARNESS_HF_ENV.split(",") if x.strip()]:
        if not m.startswith("hf:"):
            m = f"hf:{m}"
        ENGINE_CONFIG['CONSTANTS']['HARNESS_GENERATOR_MODELS'].append(m)

# ------------------------------------------------------------------------------------
# Prompt templates (unchanged public interface; improved robustness inside)
# ------------------------------------------------------------------------------------
META_PROMPT_TEMPLATE = r"""
You are an expert Test Driven Development (TDD) engineer. Your task is to generate a Python script containing a test harness for a given algorithmic task description.

The generated script MUST contain:
1.  A dictionary named `TASK_CONSTANTS`. It must include `EXEC_TIMEOUT` (in seconds, estimate a reasonable value based on the task description, e.g., 10s for simple poly-time, 30s+ for search/BFS on small inputs) and `ERROR_TIMEOUT` (a string message).
2.  A function `test_code(code: str, task_config: Dict) -> Tuple[bool, str, Optional[Dict]]`. This function MUST be self-contained *except* for standard libraries (json, sys, subprocess, tempfile, os, psutil, traceback, time, typing, random, queue, collections).

The `test_code` function must:
-   Accept the Python `code` as a string and `task_config` (which will be the `TASK_CONSTANTS` dict).
-   Define "ground truth" logic based *only* on the task description.
-   Define a list of test cases including edge cases and randomized small cases.
-   Write the `code` string to a temporary file.
-   Run the temporary file as a subprocess (`sys.executable`) for each test case, passing the test input as a JSON string in `sys.argv[1]`.
-   Use `subprocess.communicate` with the `EXEC_TIMEOUT` from `task_config`.
-   Parse the subprocess `stdout` as JSON.
-   Verify the JSON output structure and correctness against ground truth.
-   Return `(True, "All tests passed", summary_dict)` on success.
-   Return `(False, "Error message", summary_dict)` on failure (timeout, JSON error, logic error, etc.).

TASK DESCRIPTION:
---
{task_prompt}
---

Your response MUST be *only* the raw, executable Python code containing `TASK_CONSTANTS` and `test_code`. Do not include explanations, markdowns, or any other text.
"""

FIX_PROMPT_TEMPLATE = r"""
You are a Python debugging assistant. The following code, intended to solve the task below, did not work correctly. Fix it to meet all requirements.

--- TASK ---
{task_prompt}
---

--- FAILED CODE ---
{code}
---

--- ISSUE ---
{error}
---

Respond with *only* the fixed, self-contained, executable Python code. Do not include explanations or markdown.
"""

REFACTOR_PROMPT_TEMPLATE = r"""
You are an expert Python programmer. Compare the current and previous versions of this code and perform a full refactor to improve efficiency and correctness, ensuring it solves the task.

--- TASK ---
{task_prompt}
---

--- CURRENT CODE ---
{code}
---

--- PREVIOUS VERSION ---
{prev}
---

Respond with *only* the refactored, self-contained, executable Python code. Do not include explanations or markdown.
"""

REFACTOR_NO_PREV_TEMPLATE = r"""
You are an expert Python programmer. Refactor the following code to improve its efficiency and correctness, ensuring it solves the task.

--- TASK ---
{task_prompt}
---

--- CURRENT CODE ---
{code}
---

Respond with *only* the refactored, self-contained, executable Python code. Do not include explanations or markdown.
"""

# ------------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------------
def clean_code(code: str) -> str:
    """Strip JSON and markdown fences from LLM code responses."""
    try:
        data = json.loads(code)
        if isinstance(data, dict) and 'choices' in data and data['choices']:
            content = data['choices'][0].get('message', {}).get('content', '')
            if isinstance(content, str):
                code = content
    except Exception:
        pass
    m = re.search(r'```python\s*(.*?)\s*```', code, re.DOTALL | re.MULTILINE)
    if m:
        code = m.group(1)
    code = re.sub(r'^```python\n?', '', code, flags=re.MULTILINE)
    code = re.sub(r'\n?```\s*$', '', code, flags=re.MULTILINE)
    code = code.strip()
    # filter obvious garbage
    if ("discord.gg" in code) and len([l for l in code.splitlines() if l.strip() and not l.strip().startswith("#")]) < 3:
        return ""
    return code

def generate_prompt_templates(initial_prompt: str) -> Dict[str, str]:
    return {
        'INITIAL': initial_prompt,
        'TASK_PROMPT': initial_prompt,
        'FIX': FIX_PROMPT_TEMPLATE,
        'REFACTOR': REFACTOR_PROMPT_TEMPLATE,
        'REFACTOR_NO_PREV': REFACTOR_NO_PREV_TEMPLATE
    }

def save_json(obj: Any, folder: str, filename: str) -> None:
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def append_log_line(folder: str, model_name: str, line: str) -> None:
    os.makedirs(folder, exist_ok=True)
    safe = re.sub(r'[^a-zA-Z0-9_.-]+', '_', model_name) + ".log"
    path = os.path.join(folder, safe)
    with open(path, "a", encoding="utf-8") as f:
        f.write(f"{datetime.now().isoformat()} | {line}\n")

def flush_queue_to_logs(q: queue.Queue, logs_folder: str) -> None:
    try:
        while True:
            model, kind, message = q.get_nowait()
            if kind in ("log", "status"):
                append_log_line(logs_folder, model, f"[{kind}] {message}")
    except queue.Empty:
        return

def get_models_list(config: Dict) -> List[str]:
    """
    Build a model list from the public g4f list + live working list + optional HF models.
    Supports env var HF_MODELS (comma-separated), where entries can be bare or start with 'hf:'.
    """
    working_models = set()
    url_txt = config['URLS']['WORKING_RESULTS']
    try:
        resp = requests.get(url_txt, timeout=config['CONSTANTS']['REQUEST_TIMEOUT'])
        resp.raise_for_status()
        for line in resp.text.splitlines():
            if config['CONSTANTS']['DELIMITER_MODEL'] in line:
                parts = [p.strip() for p in line.split(config['CONSTANTS']['DELIMITER_MODEL'])]
                if len(parts) == 3 and parts[2] == config['CONSTANTS']['MODEL_TYPE_TEXT']:
                    m = parts[1]
                    if not any(x in m.lower() for x in ["flux", "image", "vision", "audio", "video"]):
                        working_models.add(m)
    except Exception as e:
        print(f"Warning: failed to download working models list: {e}", file=sys.stderr)

    try:
        from g4f.models import Model as G4FModel
        g4f_models = set([m for m in G4FModel.__all__() if not any(x in m.lower() for x in ["flux","image","vision","audio","video"])])
    except Exception:
        g4f_models = set()

    all_models = set()
    all_models |= working_models
    all_models |= g4f_models

    # add HF custom models
    for m in config.get('CUSTOM_MODELS', {}).get('HF_MODELS', []):
        all_models.add(m)

    # simple pruning of obvious aliases
    banned = {'sldx-turbo', 'turbo'}
    return [m for m in sorted(all_models) if m not in banned]

def llm_query(model: Any, prompt: str, retries: Dict, config: Dict, progress_q: queue.Queue, stage: str = None) -> Optional[str]:
    """Query model (supports g4f models, names, and 'hf:<repo>' via Provider.HuggingFace)."""
    if hasattr(model, "name"):
        model_name = model.name
    else:
        model_name = str(model)
    _local.model_name = model_name
    _local.queue = progress_q
    _local.data = {'tried': [], 'errors': {}, 'success': None, 'model': model_name}
    _local.stage = stage

    timeout = config['CONSTANTS']['REQUEST_TIMEOUT']
    is_hf = model_name.startswith("hf:")
    token = config['CUSTOM_MODELS'].get('HF_API_TOKEN')
    host = config['CUSTOM_MODELS'].get('HF_API_URL')

    for attempt in range(retries['max_retries'] + 1):
        try:
            if is_hf:
                repo = model_name.split(":",1)[1]
                if not token:
                    progress_q.put((model_name, 'log', 'HF_API_TOKEN is not set; skipping HuggingFace call.'))
                    return None
                base = host or "https://api-inference.huggingface.co"
                url = base.rstrip("/") + "/models/" + repo
                headers = {"Authorization": f"Bearer {token}"}
                payload = {
                    "inputs": prompt,
                    "parameters": {"max_new_tokens": 768, "return_full_text": False},
                    "options": {"wait_for_model": True}
                }
                try:
                    r = requests.post(url, headers=headers, json=payload, timeout=timeout*2)
                    if r.status_code == 200:
                        data = r.json()
                        # common HF responses: list[{'generated_text': ...}] or dict with 'generated_text'
                        if isinstance(data, list) and data and isinstance(data[0], dict) and 'generated_text' in data[0]:
                            resp = data[0]['generated_text']
                        elif isinstance(data, dict) and 'generated_text' in data:
                            resp = data['generated_text']
                        elif isinstance(data, dict) and 'error' in data:
                            progress_q.put((model_name, 'log', f"HF error: {data['error']}"))
                            resp = None
                        else:
                            # fallback to raw text if present
                            resp = r.text
                    else:
                        progress_q.put((model_name, 'log', f"HF HTTP {r.status_code}: {r.text[:200]}"))
                        resp = None
                except Exception as e:
                    progress_q.put((model_name, 'log', f"HF request error: {e}"))
                    resp = None
            else:
                resp = g4f.ChatCompletion.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    provider=Provider.AnyProvider,
                    timeout=timeout
                )
            if resp and str(resp).strip():
                progress_q.put((model_name, 'log', f"Received response at stage '{stage}' (len={len(str(resp))})."))
                return str(resp).strip()
        except ModelNotFoundError as e:
            progress_q.put((model_name, 'log', f"ModelNotFoundError: {e}"))
            return None
        except Exception as e:
            progress_q.put((model_name, 'log', f"Error (attempt {attempt+1}): {e}"))
        if attempt < retries['max_retries']:
            time.sleep(retries['backoff_factor'] * (2 ** attempt))
    return None

def find_working_harness_model(config: Dict, progress_q: queue.Queue) -> Optional[Any]:
    """Ping models in HARNESS_GENERATOR_MODELS; return the first that replies 'OK'."""
    test_prompt = "Respond with only the single word: 'OK'"
    for m in config['CONSTANTS']['HARNESS_GENERATOR_MODELS']:
        name = m.name if hasattr(m, "name") else str(m)
        progress_q.put(("HARNESS_TEST", 'log', f"Pinging model: {name}"))
        resp = llm_query(m, test_prompt, config['RETRIES']['HARNESS_TEST'], config, progress_q, 'harness_test')
        if resp and resp.strip().lower() == "ok":
            progress_q.put(("HARNESS_TEST", 'log', f"SUCCESS: Selected {name}"))
            return m
        progress_q.put(("HARNESS_TEST", 'log', f"FAILED: {name}"))
    return None

def generate_task_harness(initial_prompt: str, harness_model: Any, engine_config: Dict, progress_q: queue.Queue) -> Optional[Dict]:
    """Ask the validated model to produce TASK_CONSTANTS + test_code() tailored to the algorithmic task prompt."""
    meta_prompt = META_PROMPT_TEMPLATE.format(task_prompt=initial_prompt)
    resp = llm_query(harness_model, meta_prompt, engine_config['RETRIES']['HARNESS_GEN'], engine_config, progress_q, 'generate_harness')
    if not resp:
        progress_q.put(("HARNESS_GEN", 'log', "FATAL: Empty harness code response."))
        return None
    code = clean_code(resp)
    if not code:
        progress_q.put(("HARNESS_GEN", 'log', "FATAL: Harness code empty after cleaning."))
        return None
    # execute generated code in a safe-ish namespace
    ctx: Dict[str, Any] = {}
    try:
        exec(code, {
            'json': json, 'sys': sys, 'subprocess': subprocess, 'tempfile': tempfile, 'os': os,
            'psutil': psutil, 'traceback': traceback, 'time': time, 'perf_counter': perf_counter,
            'Dict': Dict, 'List': List, 'Optional': Optional, 'Tuple': Tuple, 'random': random,
            'queue': queue, 're': re, 'collections': __import__('collections'),
        }, ctx)
    except Exception as e:
        progress_q.put(("HARNESS_GEN", 'log', f"FATAL: exec generated harness failed: {e}"))
        progress_q.put(("HARNESS_GEN", 'log', f"CODE_SNIPPET_START\n{code[:2000]}\nCODE_SNIPPET_END"))
        return None
    if 'test_code' not in ctx or not callable(ctx['test_code']):
        progress_q.put(("HARNESS_GEN", 'log', "FATAL: test_code() missing or not callable."))
        return None
    if 'TASK_CONSTANTS' not in ctx or not isinstance(ctx['TASK_CONSTANTS'], dict):
        progress_q.put(("HARNESS_GEN", 'log', "FATAL: TASK_CONSTANTS missing or not a dict."))
        return None
    return {
        "test_code_func": ctx['test_code'],
        "TASK_CONSTANTS": ctx['TASK_CONSTANTS'],
        "generated_source_code": code,
    }

def process_model(model: str, task_config: Dict, prompts: Dict, engine_config: Dict, progress_q: queue.Queue) -> Dict:
    """Main per-model pipeline: initial -> test/fix -> refactor -> loops -> final test (interfaces unchanged)."""
    iterations = []
    current_code = None
    prev_code = None
    early_stop = False

    TASK_CONSTANTS = task_config['TASK_CONSTANTS']
    test_code_func = task_config['test_code_func']

    STAGES = engine_config['STAGES']
    RETRIES = engine_config['RETRIES']
    CONSTANTS = engine_config['CONSTANTS']
    task_prompt = prompts.get('TASK_PROMPT', '')

    num_loops = int(CONSTANTS['NUM_REFACTOR_LOOPS'])
    total_stages = 1 + 1 + 1 + 1 + (num_loops * 2) + 1
    done = 0
    def progress(stage):
        nonlocal done
        done += 1
        progress_q.put((model, 'status', f'Stage: {stage} ({done}/{total_stages})'))
        progress_q.put((model, 'log', f'Enter stage: {stage}'))

    def test(code, stage_name):
        if not code or not code.strip():
            return False, "No code to test", None
        try:
            ok, msg, summary = test_code_func(code, TASK_CONSTANTS)
            return ok, msg, summary
        except Exception as e:
            tb = traceback.format_exc()
            return False, f"Exception in test_code(): {e}", {'error': str(e), 'traceback': tb}

    def ask_llm(prompt_text, stage_name, retries_key='FIX'):
        rconf = RETRIES[retries_key]
        resp = llm_query(model, prompt_text, rconf, engine_config, progress_q, stage_name)
        tried = getattr(_local, "data", {}).get('tried', [])
        succ = getattr(_local, "data", {}).get('success', None)
        if resp:
            cleaned = clean_code(resp)
            return cleaned, None, tried, succ
        else:
            return None, CONSTANTS['ERROR_NO_RESPONSE'], tried, succ

    def add_iter(stage, response, error, test_summary, tried, success_p):
        iterations.append({
            'providers_tried': tried,
            'success_provider': success_p,
            'stage': stage,
            'response': response,
            'error': error,
            'test_summary': test_summary
        })

    # 1) Initial code draft
    stage = STAGES['INITIAL']; progress(stage)
    prompt = prompts['INITIAL']
    current_code, err, tried, sprov = ask_llm(prompt, stage, 'INITIAL')
    add_iter(stage, current_code, err, None, tried, sprov)
    if err:
        return {'model': model, 'iterations': iterations, 'final_code': None,
                'final_test': {'success': False, 'summary': None, 'issue': 'No initial response'}}

    # 2) Test + Fix initial
    stage = STAGES['FIX_INITIAL']; progress(stage)
    ok, issue, summary = test(current_code, stage)
    if not ok:
        prompt = prompts['FIX'].format(task_prompt=task_prompt, code=current_code, error=str(issue).replace('{','{{').replace('}','}}'))
        current_code, err, tried, sprov = ask_llm(prompt, stage, 'FIX')
        add_iter(stage, current_code, err, summary, tried, sprov)
        if err:
            early_stop = True
    else:
        add_iter(stage, current_code, None, summary, [], None)
    if early_stop:
        return {'model': model, 'iterations': iterations, 'final_code': current_code,
                'final_test': {'success': False, 'summary': summary, 'issue': f'LLM error during {stage}'}}

    # 3) First refactor (no prev)
    prev_code = current_code
    stage = STAGES['REFACTOR_FIRST']; progress(stage)
    prompt = prompts['REFACTOR_NO_PREV'].format(task_prompt=task_prompt, code=current_code)
    current_code, err, tried, sprov = ask_llm(prompt, stage, 'INITIAL')
    add_iter(stage, current_code, err, None, tried, sprov)
    if err:
        current_code = prev_code

    # 4) Test + Fix after first refactor
    stage = STAGES['FIX_AFTER_REFACTOR']; progress(stage)
    ok, issue, summary = test(current_code, stage)
    if not ok:
        prompt = prompts['FIX'].format(task_prompt=task_prompt, code=current_code, error=str(issue).replace('{','{{').replace('}','}}'))
        current_code, err, tried, sprov = ask_llm(prompt, stage, 'FIX')
        add_iter(stage, current_code, err, summary, tried, sprov)
        if err:
            early_stop = True
    else:
        add_iter(stage, current_code, None, summary, [], None)
    if early_stop:
        return {'model': model, 'iterations': iterations, 'final_code': current_code,
                'final_test': {'success': False, 'summary': summary, 'issue': f'LLM error during {stage}'}}

    # 5) Refactor loops
    for i in range(num_loops):
        if not current_code:
            progress(f'loop {i+1} (skip)'); progress(f'loop {i+1} fix (skip)')
            continue
        # 5a refactor
        stage = f"{engine_config['STAGES']['REFACTOR']}_{i+1}"; progress(stage)
        prompt = prompts['REFACTOR'].format(task_prompt=task_prompt, code=current_code, prev=prev_code)
        prev_code = current_code
        current_code, err, tried, sprov = ask_llm(prompt, stage, 'INITIAL')
        add_iter(stage, current_code, err, None, tried, sprov)
        if err:
            current_code = prev_code
        # 5b test & fix
        stage = f"{engine_config['STAGES']['FIX_LOOP']}_{i+1}"; progress(stage)
        ok, issue, summary = test(current_code, stage)
        if not ok:
            prompt = prompts['FIX'].format(task_prompt=task_prompt, code=current_code, error=str(issue).replace('{','{{').replace('}','}}'))
            current_code, err, tried, sprov = ask_llm(prompt, stage, 'FIX')
            add_iter(stage, current_code, err, summary, tried, sprov)
            if err:
                break
        else:
            add_iter(stage, current_code, None, summary, [], None)

    # 6) Final test
    stage = 'final_test'; progress(stage)
    ok, issue, summary = test(current_code, stage)
    add_iter(stage, current_code, None if ok else issue, summary, [], None)
    return {
        'model': model,
        'iterations': iterations,
        'final_code': current_code,
        'final_test': {'success': ok, 'summary': summary, 'issue': issue}
    }

def main():
    if len(sys.argv) < 2:
        print("Usage: python universal_tester.py <path_to_task_prompt.txt>", file=sys.stderr)
        sys.exit(1)

    task_prompt_path = sys.argv[1]
    try:
        with open(task_prompt_path, "r", encoding="utf-8") as f:
            initial_prompt = f.read()
        if not initial_prompt.strip():
            print(f"Error: Task prompt is empty: {task_prompt_path}", file=sys.stderr)
            sys.exit(1)
        print(f"Loaded task prompt: {task_prompt_path}")
    except Exception as e:
        print(f"Error reading prompt: {e}", file=sys.stderr)
        sys.exit(1)

    progress_q: "queue.Queue[Tuple[str,str,str]]" = queue.Queue()

    # 1) Harness model
    harness_model = find_working_harness_model(ENGINE_CONFIG, progress_q)
    if harness_model is None:
        flush_queue_to_logs(progress_q, os.path.join(RESULTS_DIR, "logs"))
        print("FATAL: No working model to generate harness.", file=sys.stderr)
        sys.exit(1)

    # 2) Generate harness
    task_config = generate_task_harness(initial_prompt, harness_model, ENGINE_CONFIG, progress_q)
    if task_config is None:
        flush_queue_to_logs(progress_q, os.path.join(RESULTS_DIR, "logs"))
        print("FATAL: Could not generate test harness.", file=sys.stderr)
        sys.exit(1)

    # Save harness
    os.makedirs(RESULTS_DIR, exist_ok=True)
    harness_path = os.path.join(RESULTS_DIR, "__generated_test_harness.py")
    with open(harness_path, "w", encoding="utf-8") as f:
        f.write("# Auto-generated by universal_tester.py\n")
        f.write(f"# Generated at: {datetime.now().isoformat()}\n\n")
        f.write(task_config.get("generated_source_code",""))
    print(f"Harness saved to: {harness_path}")

    prompts = generate_prompt_templates(initial_prompt)
    print("Prompt templates prepared.")

    # 3) Model list
    if ENGINE_CONFIG.get('CUSTOM_MODELS', {}).get('HF_MODELS') and not ENGINE_CONFIG['CUSTOM_MODELS'].get('HF_API_TOKEN'):
        print("Warning: HF models are configured but HF_API_TOKEN is not set — they will likely fail.", file=sys.stderr)
    try:
        models = get_models_list(ENGINE_CONFIG)
        if not models:
            print("No models found. Exiting.", file=sys.stderr)
            sys.exit(1)
    except Exception as e:
        print(f"Failed to get model list: {e}", file=sys.stderr)
        sys.exit(1)

    if MAX_MODELS_TO_TEST > 0:
        models = models[:MAX_MODELS_TO_TEST]

    all_results: Dict[str, Any] = {}
    intermediates_dir = RESULTS_DIR
    logs_folder = os.path.join(RESULTS_DIR, "logs")
    os.makedirs(intermediates_dir, exist_ok=True)
    os.makedirs(logs_folder, exist_ok=True)

    started = datetime.now().isoformat()
    print(f"--- STARTING TEST on {len(models)} models ---")

    try:
        with ThreadPoolExecutor(max_workers=ENGINE_CONFIG['CONSTANTS']['MAX_WORKERS']) as ex:
            futures = {ex.submit(process_model, m, task_config, prompts, ENGINE_CONFIG, progress_q): m for m in models}
            counter = 0
            t0 = perf_counter()
            for fut in as_completed(futures):
                model = futures[fut]
                counter += 1
                try:
                    res = fut.result()
                    all_results[model] = res
                    status = "SUCCESS" if res.get('final_test',{}).get('success', False) else "FAILED"
                    # persist final code per model for inspection
                    if res.get('final_code'):
                        safe = re.sub(r'[^a-zA-Z0-9_.-]+', '_', model)
                        with open(os.path.join(intermediates_dir, f"{safe}_final.py"), "w", encoding="utf-8") as f:
                            f.write(res['final_code'])
                    print(f"[{counter}/{len(futures)}] {model}: {status}")
                except Exception as e:
                    tb = traceback.format_exc()
                    all_results[model] = {'error': str(e), 'traceback': tb, 'iterations': [], 'final_code': None,
                                          'final_test': {'success': False, 'summary': None, 'issue': str(e)}}
                finally:
                    flush_queue_to_logs(progress_q, logs_folder)
                if (counter % ENGINE_CONFIG['CONSTANTS']['N_SAVE'] == 0) or (counter == len(futures)):
                    save_json(all_results, intermediates_dir, f"intermediate_results_{counter}.json")
    except KeyboardInterrupt:
        print("\nKeyboardInterrupt: finishing current threads.")

    t1 = perf_counter()
    finished = datetime.now().isoformat()
    total_sec = t1 - t0

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_name = f"final_results_{timestamp}.json"
    save_json(all_results, intermediates_dir, final_name)
    meta = {
        "started_at": started,
        "finished_at": finished,
        "total_seconds": total_sec,
        "seed": SEED,
        "env": {
            "UT_MAX_MODELS": os.environ.get("UT_MAX_MODELS", ""),
            "UT_NUM_REFACTOR_LOOPS": os.environ.get("UT_NUM_REFACTOR_LOOPS", ""),
            "UT_MAX_WORKERS": os.environ.get("UT_MAX_WORKERS", ""),
            "HF_MODELS": os.environ.get("HF_MODELS", ""),
            "HF_API_URL": os.environ.get("HF_API_URL", ""),
            "HF_API_TOKEN_set": bool(os.environ.get("HF_API_TOKEN"))
        },
        "final_results_file": final_name
    }
    save_json(meta, intermediates_dir, f"run_manifest_{timestamp}.json")
    print(f"Final results: {os.path.join(intermediates_dir, final_name)}")
    print(f"Totals: {sum(1 for r in all_results.values() if r.get('final_test',{}).get('success'))} SUCCESS, "
          f"{len(all_results) - sum(1 for r in all_results.values() if r.get('final_test',{}).get('success'))} FAILED.")
    print(f"Total execution time: {total_sec:.2f} sec.")
    flush_queue_to_logs(progress_q, logs_folder)

if __name__ == "__main__":
    main()

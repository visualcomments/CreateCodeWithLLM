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

<<<<<<< HEAD
class TrackedRotated(OriginalRotatedProvider):  # type: ignore
=======
import threading
local = threading.local()

from g4f.errors import ModelNotFoundError
import queue

def clean_code(code: str) -> str:
    """
    Cleans code from markdown wrappers like ```python ... ``` and JSON metadata (OpenAI-like).
    First, checks for JSON, extracts content from choices[0].message.content if possible.
    If JSON doesn't parse, looks for a markdown block in the string and extracts its content.
    Then removes lines with ```python, ``` and extra empty lines.
    """
    original_len = len(code)
    
    # Step 1: Check for JSON wrapper (OpenAI-style)
    content_from_json = None
    try:
        data = json.loads(code)
        if isinstance(data, dict) and 'choices' in data and len(data['choices']) > 0:
            content = data['choices'][0].get('message', {}).get('content', '')
            if isinstance(content, str):
                content_from_json = content
                code = content  # Replace with content for further cleaning
    except (json.JSONDecodeError, KeyError, IndexError, TypeError):
        pass  # Not JSON — continue

    # Step 2: If JSON didn't work or content_from_json is empty, search for markdown block in original
    # Find first block ```python\n... (up to next ``` or end)
    
    # === FIX 1 (Was: match = re.search(r'...) ===
    match = re.search(r'```(?:python\n)?(.*?)\n```', code, re.DOTALL | re.MULTILINE)

    if match:
        code = match.group(1)
    # Alternative: if block without closing ```, search from first ``` to end
    else:
        # === FIX 2 (Was: match = re.search(r'...) ===
        match = re.search(r'```(?:python\n)?(.*?)$', code, re.DOTALL | re.MULTILINE)

        if match:
            code = match.group(1)

    # Step 3: Final regex cleanup (for nested markdown)
    # Remove ```python block at start
    # === FIX 3 (Was: code = re.sub(r'^...) ===
    code = re.sub(r'^```(?:python\n)?', '', code, flags=re.MULTILINE)

    # Remove ``` block at end
    code = re.sub(r'\n?```\s*$', '', code, flags=re.MULTILINE)
    # Remove extra newlines at start and end
    code = re.sub(r'^\n+', '', code, flags=re.MULTILINE)
    code = re.sub(r'\n+$', '\n', code, flags=re.MULTILINE)
    
    cleaned = code.strip()
    
    return cleaned

# Custom Rotated with tracking (patching create_async_generator, logs in loop)
class TrackedRotated(OriginalRotatedProvider):
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

<<<<<<< HEAD
# ------------------------------------------------------------------------------------
# Engine configuration (interfaces unchanged)
# ------------------------------------------------------------------------------------
ENGINE_CONFIG: Dict[str, Any] = {
=======
except AttributeError:
     print("Failed to apply Monkey-patch for g4f.debug (attributes not found)", file=sys.stderr)


try:
    import g4f.providers.retry_provider as retry_mod
    OriginalRotatedProvider = retry_mod.RotatedProvider
except ImportError:
    print("Failed to import g4f.providers.retry_provider. Using fallback.", file=sys.stderr)
    class OriginalRotatedProvider:
        pass

import g4f
from g4f import Provider
import threading
local = threading.local()
from g4f.errors import ModelNotFoundError
import queue

def clean_code(code: str) -> str:
    """
    Cleans code from markdown wrappers like ```python ... ``` and JSON metadata.
    """
    original_len = len(code)
    content_from_json = None
    try:
        data = json.loads(code)
        if isinstance(data, dict) and 'choices' in data and len(data['choices']) > 0:
            content = data['choices'][0].get('message', {}).get('content', '')
            if isinstance(content, str):
                content_from_json = content
                code = content
    except (json.JSONDecodeError, KeyError, IndexError, TypeError):
        pass

    match = re.search(r'```(?:python\n)?(.*?)\n```', code, re.DOTALL | re.MULTILINE)
    if match:
        code = match.group(1)
    else:
        match = re.search(r'```(?:python\n)?(.*?)$', code, re.DOTALL | re.MULTILINE)
        if match:
            code = match.group(1)

    code = re.sub(r'^```(?:python\n)?', '', code, flags=re.MULTILINE)
    code = re.sub(r'\n?```\s*$', '', code, flags=re.MULTILINE)
    code = re.sub(r'^\n+', '', code, flags=re.MULTILINE)
    code = re.sub(r'\n+$', '\n', code, flags=re.MULTILINE)
    cleaned = code.strip()
    return cleaned

class TrackedRotated(OriginalRotatedProvider):
    """
    Custom RotatedProvider to track which provider is being used and log to queue.
    """
    async def create_async_generator(self, model, messages, **kwargs):
        if not hasattr(local, 'current_data') or local.current_data is None:
            local.current_data = {'tried': [], 'errors': {}, 'success': None, 'model': model}
        current_data = local.current_data
        current_data['tried'] = []
        current_data['errors'] = {}
        current_data['success'] = None
        current_data['model'] = model
        if hasattr(local, 'current_model') and hasattr(local, 'current_queue') and self.providers:
            local.current_queue.put((local.current_model, 'log', f'1) Found providers: {[p.__name__ for p in self.providers]}'))
            local.current_queue.put((local.current_model, 'log', f'Debug: TrackedRotated called for model {model}'))
        
        if not self.providers:
             raise ModelNotFoundError(f"No providers found for model {model}", [])

        for provider_class in self.providers:
            p = None
            if isinstance(provider_class, str):
                provider_name = provider_class
            else:
                provider_name = provider_class.__name__ if hasattr(provider_class, '__name__') else str(provider_class)
            current_data['tried'].append(provider_name)
            if hasattr(local, 'current_model') and hasattr(local, 'current_queue'):
                local.current_queue.put((local.current_model, 'log', f'2) Trying {provider_name} with model: {model}'))
            try:
                if isinstance(provider_class, str):
                    if hasattr(Provider, provider_class):
                        provider_class = getattr(Provider, provider_class)
                    else:
                        raise ValueError(f"Provider '{provider_name}' not found in Provider")
                p = provider_class()
                async for chunk in p.create_async_generator(model, messages, **kwargs):
                    yield chunk
                if hasattr(local, 'current_model') and hasattr(local, 'current_queue'):
                    local.current_queue.put((local.current_model, 'log', f'3) Success from {provider_name}'))
                    current_data['success'] = provider_name
                return
            except Exception as e:
                error_str = str(e)
                if hasattr(local, 'current_model') and hasattr(local, 'current_queue'):
                    error_msg = f'3) Error {provider_name}: {error_str}'
                    local.current_queue.put((local.current_model, 'log', error_msg))
                current_data['errors'][provider_name] = error_str
                if p:
                    if hasattr(p, '__del__'):
                        p.__del__()
                continue
        try:
            if hasattr(local, 'current_model') and hasattr(local, 'current_queue'):
                local.current_queue.put((local.current_model, 'log', f'Debug: TrackedRotated finished, tried_providers={current_data["tried"]}'))
        except Exception:
            pass
        raise ModelNotFoundError(f"No working provider for model {model}", current_data['tried'])

try:
    retry_mod.RotatedProvider = TrackedRotated
except NameError:
    print("Failed to apply Monkey-patch for RotatedProvider (retry_mod not defined)", file=sys.stderr)

try:
    original_log = g4f.debug.log
    original_error = g4f.debug.error
    def patched_log(message, *args, **kwargs):
        message_str = str(message) if not isinstance(message, str) else message
        if hasattr(local, 'current_model') and hasattr(local, 'current_queue'):
            if 'AnyProvider: Using providers:' in message_str:
                providers_str = message_str.split('providers: ')[1].split(" for model")[0].strip("'")
                local.current_queue.put((local.current_model, 'log', f'1) Found providers: [{providers_str}]'))
            elif 'Attempting provider:' in message_str:
                provider_str = message_str.split('provider: ')[1].strip()
                local.current_queue.put((local.current_model, 'log', f'2) Trying {provider_str}'))
    def patched_error(message, *args, **kwargs):
        message_str = str(message) if not isinstance(message, str) else message
        if hasattr(local, 'current_model') and hasattr(local, 'current_queue'):
            if 'failed:' in message_str:
                fail_str = message_str.split('failed: ')[1].strip()
                local.current_queue.put((local.current_model, 'log', f'3) Error {fail_str}'))
            elif 'success' in message_str.lower():
                success_str = message_str.split('success: ')[1].strip() if 'success: ' in message_str else 'success'
                local.current_queue.put((local.current_model, 'log', f'3) Success {success_str}'))
    g4f.debug.log = patched_log
    g4f.debug.error = patched_error
except AttributeError:
     print("Failed to apply Monkey-patch for g4f.debug (attributes not found)", file=sys.stderr)



# =============================================================================
# === ENGINE CONFIGURATION ===
# =============================================================================


HARNESS_VALIDATION_CODE_CONTENT = """
import json
import sys

def solve_task(input_data):
    # Example placeholder logic: Replace this with the *actual* correct solution
    # for the specific task you are testing. This is just an example for sorting.
    if isinstance(input_data, list) and all(isinstance(x, (int, float)) for x in input_data):
        return sorted(input_data)
    # Add other logic specific to your task here
    return input_data

# Example: If the script receives an argument, parse it as JSON and run solve_task
if len(sys.argv) > 1:
    try:
        input_data = json.loads(sys.argv[1])
    except json.JSONDecodeError:
        print(json.dumps({"error": "Invalid JSON input"}), file=sys.stderr)
        sys.exit(1)
else:
    # Provide a default input if no argument is given (optional, for direct testing)
    input_data = [3, 1, 4, 1, 5]

# Execute the task
result = solve_task(input_data)

# Print the result as JSON
print(json.dumps(result))
"""


ENGINE_CONFIG = {
>>>>>>> 378884a05e1a0098f1c86c0a0cb10d3b3c2af67e
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
<<<<<<< HEAD
        # Add huggingface chat models by prefixing with "hf:"
        # e.g., "hf:meta-llama/Llama-3.1-8B-Instruct"
        'HF_MODELS': [],
=======
        'HF_MODELS': [
            "hf:meta-llama/Llama-2-7b-chat-hf",
        ],
>>>>>>> 378884a05e1a0098f1c86c0a0cb10d3b3c2af67e
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
<<<<<<< HEAD
        'NUM_REFACTOR_LOOPS': DEFAULT_REFACTOR_LOOPS,
        'INTERMEDIATE_FOLDER': RESULTS_DIR,
        'HARNESS_GENERATOR_MODELS': [
            # This list is probed first to generate the task-specific test harness
            # You can add HuggingFace entries using "hf:..." via env UT_HARNESS_HF (comma-separated)
            # or by editing this list.
=======
        'NUM_REFACTOR_LOOPS': 3,
        'INTERMEDIATE_FOLDER': 'results',
        'HARNESS_GENERATOR_MODELS': [ # List of models to try for harness generation
>>>>>>> 378884a05e1a0098f1c86c0a0cb10d3b3c2af67e
            g4f.models.gpt_4,

        ],
        'HARNESS_VALIDATION_CODE': HARNESS_VALIDATION_CODE_CONTENT,
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
                resp = g4f.ChatCompletion.create(
                    model=repo,
                    messages=[{"role": "user", "content": prompt}],
                    provider=Provider.HuggingFace,
                    auth=token,
                    api_host=host,
                    timeout=timeout * 2
                )
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
<<<<<<< HEAD
            progress_q.put((model_name, 'log', f"Error (attempt {attempt+1}): {e}"))
        if attempt < retries['max_retries']:
            time.sleep(retries['backoff_factor'] * (2 ** attempt))
=======
            error_msg = f'g4f Error (attempt {attempt+1}): {e}'
            progress_queue.put((model_name_str, 'log', error_msg))
            if is_hf_model:
                local.current_data['errors']['HuggingFace'] = str(e)
            
        
        if attempt < retries_config['max_retries']:
            time.sleep(retries_config['backoff_factor'] * (2 ** attempt))

>>>>>>> 378884a05e1a0098f1c86c0a0cb10d3b3c2af67e
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

<<<<<<< HEAD
def generate_task_harness(initial_prompt: str, harness_model: Any, engine_config: Dict, progress_q: queue.Queue) -> Optional[Dict]:
    """Ask the validated model to produce TASK_CONSTANTS + test_code() tailored to the algorithmic task prompt."""
=======
def generate_prompt_templates(initial_prompt: str) -> Dict[str, str]:
    """
    Generates all prompt variants based on the single initial prompt.
    """
    return {
        'INITIAL': initial_prompt,
        'FIX': FIX_PROMPT_TEMPLATE.format(task_prompt=initial_prompt, code="{code}", error="{error}"),
        'REFACTOR': REFACTOR_PROMPT_TEMPLATE.format(task_prompt=initial_prompt, code="{code}", prev="{prev}"),
        'REFACTOR_NO_PREV': REFACTOR_NO_PREV_TEMPLATE.format(task_prompt=initial_prompt, code="{code}")
    }

def generate_task_harness(initial_prompt: str, harness_model: Any, engine_config: Dict, progress_queue: queue.Queue) -> Optional[Dict]:
    """
    Uses the validated LLM to generate the `test_code` function and `TASK_CONSTANTS`.
    """
    print("--- Starting Task Harness Generation (Meta-Step) ---")
    
>>>>>>> 378884a05e1a0098f1c86c0a0cb10d3b3c2af67e
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
<<<<<<< HEAD
        "test_code_func": ctx['test_code'],
        "TASK_CONSTANTS": ctx['TASK_CONSTANTS'],
        "generated_source_code": code,
=======
        "test_code_func": context['test_code'],
        "TASK_CONSTANTS": context['TASK_CONSTANTS']
>>>>>>> 378884a05e1a0098f1c86c0a0cb10d3b3c2af67e
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
<<<<<<< HEAD
    task_prompt = prompts.get('TASK_PROMPT', '')
=======
>>>>>>> 378884a05e1a0098f1c86c0a0cb10d3b3c2af67e

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

<<<<<<< HEAD
    # 2) Test + Fix initial
    stage = STAGES['FIX_INITIAL']; progress(stage)
    ok, issue, summary = test(current_code, stage)
    if not ok:
        prompt = prompts['FIX'].format(task_prompt=task_prompt, code=current_code, error=str(issue).replace('{','{{').replace('}','}}'))
        current_code, err, tried, sprov = ask_llm(prompt, stage, 'FIX')
        add_iter(stage, current_code, err, summary, tried, sprov)
        if err:
=======
    # 2. Test & Fix (Initial)
    stage = STAGES['FIX_INITIAL']
    update_progress(stage)
    success, issue, summary = run_test(current_code, stage)
    if not success:
        prompt = prompts['FIX'].format(code=current_code, error=issue)
        current_code, llm_error, tried, s_provider = run_llm_query(prompt, stage)
        add_iteration(stage, current_code, llm_error, summary, tried, s_provider)
        if llm_error:
>>>>>>> 378884a05e1a0098f1c86c0a0cb10d3b3c2af67e
            early_stop = True
    else:
        add_iter(stage, current_code, None, summary, [], None)
    if early_stop:
        return {'model': model, 'iterations': iterations, 'final_code': current_code,
                'final_test': {'success': False, 'summary': summary, 'issue': f'LLM error during {stage}'}}

    # 3) First refactor (no prev)
    prev_code = current_code
<<<<<<< HEAD
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
=======
    stage = STAGES['REFACTOR_FIRST']
    update_progress(stage)
    prompt = prompts['REFACTOR_NO_PREV'].format(code=current_code)
    current_code, llm_error, tried, s_provider = run_llm_query(prompt, stage, 'INITIAL')
    add_iteration(stage, current_code, llm_error, None, tried, s_provider)
    if llm_error:
        current_code = prev_code
        progress_queue.put((model, 'log', f'Error {stage}, rolling back to previous code version.'))
    
    # 4. Test & Fix (After Refactor 1)
    stage = STAGES['FIX_AFTER_REFACTOR']
    update_progress(stage)
    success, issue, summary = run_test(current_code, stage)
    if not success:
        prompt = prompts['FIX'].format(code=current_code, error=issue)
        current_code, llm_error, tried, s_provider = run_llm_query(prompt, stage)
        add_iteration(stage, current_code, llm_error, summary, tried, s_provider)
        if llm_error:
>>>>>>> 378884a05e1a0098f1c86c0a0cb10d3b3c2af67e
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
<<<<<<< HEAD
        # 5a refactor
        stage = f"{ENGINE_CONFIG['STAGES']['REFACTOR']}_{i+1}"; progress(stage)
        prompt = prompts['REFACTOR'].format(task_prompt=task_prompt, code=current_code, prev=prev_code)
        prev_code = current_code
        current_code, err, tried, sprov = ask_llm(prompt, stage, 'INITIAL')
        add_iter(stage, current_code, err, None, tried, sprov)
        if err:
            current_code = prev_code
        # 5b test & fix
        stage = f"{ENGINE_CONFIG['STAGES']['FIX_LOOP']}_{i+1}"; progress(stage)
        ok, issue, summary = test(current_code, stage)
        if not ok:
            prompt = prompts['FIX'].format(task_prompt=task_prompt, code=current_code, error=str(issue).replace('{','{{').replace('}','}}'))
            current_code, err, tried, sprov = ask_llm(prompt, stage, 'FIX')
            add_iter(stage, current_code, err, summary, tried, sprov)
            if err:
=======
        
        # 5a. Refactor
        stage = f"{STAGES['REFACTOR']}_{i+1}"
        update_progress(stage)
        prompt = prompts['REFACTOR'].format(code=current_code, prev=prev_code)
        prev_code = current_code
        current_code, llm_error, tried, s_provider = run_llm_query(prompt, stage, 'INITIAL')
        add_iteration(stage, current_code, llm_error, None, tried, s_provider)
        if llm_error:
            current_code = prev_code
            progress_queue.put((model, 'log', f'Error {stage}, rolling back to previous code version.'))
        
        # 5b. Test & Fix
        stage = f"{STAGES['FIX_LOOP']}_{i+1}"
        update_progress(stage)
        success, issue, summary = run_test(current_code, stage)
        if not success:
            prompt = prompts['FIX'].format(code=current_code, error=issue)
            current_code, llm_error, tried, s_provider = run_llm_query(prompt, stage)
            add_iteration(stage, current_code, llm_error, summary, tried, s_provider)
            if llm_error:
                progress_queue.put((model, 'status', f'Error at stage: {stage}, STOPPING LOOP'))
>>>>>>> 378884a05e1a0098f1c86c0a0cb10d3b3c2af67e
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

<<<<<<< HEAD
def main():
    if len(sys.argv) < 2:
        print("Usage: python universal_tester.py <path_to_task_prompt.txt>", file=sys.stderr)
        sys.exit(1)

=======

def save_results(results, folder, filename):
    """Safely save results to JSON."""
    if not os.path.exists(folder):
        try:
            os.makedirs(folder)
        except OSError as e:
            print(f"Error creating folder {folder}: {e}", file=sys.stderr)
            return
    path = os.path.join(folder, filename)
    try:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Error saving file {path}: {e}", file=sys.stderr)

def validate_harness(test_code_func, task_constants, validation_code: str) -> bool:
    """
    Validates a generated harness against a known correct solution.
    The validation_code string must contain the complete, correct implementation
    for the task, including necessary imports and the main solving function.
    """
    print("--- Validating Generated Harness ---")
    # Example input for validation (this should ideally come from the task prompt
    # or be a standard test case known to work with the validation_code).
    # For this example, we assume the validation_code handles sys.argv[1] or a default.
    example_input = "[3, 1, 4, 1, 5]" # Example: Replace with relevant input for your task
    # The validation_code itself is the 'correct' code to be tested by the harness.
    # The harness will execute this string as a subprocess.
    code_to_test = validation_code

    try:
        # Call the generated test_code_func with the correct implementation string.
        # The harness should run this code, execute its logic (e.g., solve_task),
        # and verify the output against its internal ground truth or test cases.
        success, message, summary = test_code_func(code_to_test, task_constants)

        if success:
            print("--- Harness Validation SUCCESS ---")
            print(f"      Message: {message}")
            # Optional: Check summary for specific details if needed
            # if summary and summary.get('all_tests_passed', False):
            return True
        else:
            print(f"--- Harness Validation FAILED: {message} ---")
            # Optional: Print summary for debugging
            if summary:
                 print(f"      Summary: {summary}")
            return False

    except Exception as e:
        print(f"--- Harness Validation CRITICAL ERROR: {e} ---")
        traceback.print_exc()
        # If the harness itself crashes, it's definitely invalid.
        return False

def generate_multiple_harnesses(initial_prompt: str, engine_config: Dict, progress_queue: queue.Queue) -> List[Dict]:
    """
    Generates multiple test harnesses using different models and validates them.
    Returns a list of validated harness configurations {test_code_func, TASK_CONSTANTS}.
    """
    print("--- Generating Multiple Task Harnesses ---")
    harness_models = engine_config['CONSTANTS']['HARNESS_GENERATOR_MODELS']
    validated_harnesses = []
    validation_code = engine_config['CONSTANTS'].get('HARNESS_VALIDATION_CODE', "")
    if not validation_code:
        print("Warning: HARNESS_VALIDATION_CODE not provided. Skipping validation step.", file=sys.stderr)
        print("         This means potentially flawed harnesses might be used.", file=sys.stderr)

    for i, model in enumerate(harness_models):
        model_name_str = model.name if isinstance(model, g4f.models.Model) else str(model)
        print(f"--- Attempting Harness Generation with Model {i+1}/{len(harness_models)}: {model_name_str} ---")

        task_config = generate_task_harness(initial_prompt, model, engine_config, progress_queue)
        if task_config:
            # Mark the source model for potential logging later
            task_config['source_model'] = model_name_str

            if validation_code: # Only validate if a validation code is provided
                if validate_harness(task_config['test_code_func'], task_config['TASK_CONSTANTS'], validation_code):
                    validated_harnesses.append(task_config)
                    print(f"--- Harness from {model_name_str} is VALIDATED ---")
                else:
                    print(f"--- Harness from {model_name_str} FAILED validation and is discarded. ---")
                    # Optionally, save the failed harness for debugging
                    # failed_harness_path = os.path.join(
                    #     engine_config['CONSTANTS']['INTERMEDIATE_FOLDER'],
                    #     f"failed_harness_{model_name_str.replace('/', '_')}.py"
                    # )
                    # try:
                    #     import inspect
                    #     harness_code = inspect.getsource(task_config['test_code_func'])
                    #     constants_code = f"TASK_CONSTANTS = {json.dumps(task_config['TASK_CONSTANTS'], indent=4)}\n"
                    #     with open(failed_harness_path, 'w', encoding='utf-8') as f:
                    #         f.write(f"# Failed harness from {model_name_str}\n")
                    #         f.write(constants_code)
                    #         f.write(harness_code)
                    #     print(f"      Saved failed harness for inspection to: {failed_harness_path}")
                    # except Exception as e_inspect:
                    #     print(f"      Could not save failed harness: {e_inspect}", file=sys.stderr)
            else: # If no validation code, add the harness anyway (less robust)
                 validated_harnesses.append(task_config)
                 print(f"--- Harness from {model_name_str} added without validation. ---")

    print(f"--- Generated {len(validated_harnesses)} validated harness(es) out of {len(harness_models)} attempts. ---")
    return validated_harnesses

# --- Modify the main function ---
def main():
    """
    Main function: Loads task prompt from .txt, generates harness(es),
    loads models, and starts thread pool.
    """
    if len(sys.argv) != 2:
        print("Usage: python script.py <task_prompt_file.txt>", file=sys.stderr)
        sys.exit(1)
>>>>>>> 378884a05e1a0098f1c86c0a0cb10d3b3c2af67e
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

<<<<<<< HEAD
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
=======
    progress_queue = queue.Queue()

    # 1. Generate multiple validated harnesses
    harness_configs = generate_multiple_harnesses(initial_prompt, ENGINE_CONFIG, progress_queue)
    if not harness_configs:
        print("FATAL: Could not generate any validated task harnesses. Exiting.", file=sys.stderr)
        sys.exit(1)

    # Select the first validated harness for testing models (or implement consensus logic)
    # For simplicity, use the first one found.
    selected_task_config = harness_configs[0]
    print(f"Selected harness from model: {selected_task_config.get('source_model', 'Unknown')}")
>>>>>>> 378884a05e1a0098f1c86c0a0cb10d3b3c2af67e

    # Save harness
    os.makedirs(RESULTS_DIR, exist_ok=True)
    harness_path = os.path.join(RESULTS_DIR, "__generated_test_harness.py")
    with open(harness_path, "w", encoding="utf-8") as f:
        f.write("# Auto-generated by universal_tester.py\n")
        f.write(f"# Generated at: {datetime.now().isoformat()}\n\n")
        f.write(task_config.get("generated_source_code",""))
    print(f"Harness saved to: {harness_path}")

<<<<<<< HEAD
    prompts = generate_prompt_templates(initial_prompt)
    print("Prompt templates prepared.")

    # 3) Model list
    if ENGINE_CONFIG.get('CUSTOM_MODELS', {}).get('HF_MODELS') and not ENGINE_CONFIG['CUSTOM_MODELS'].get('HF_API_TOKEN'):
        print("Warning: HF models are configured but HF_API_TOKEN is not set — they will likely fail.", file=sys.stderr)
=======
    # --- Rest of the main function remains largely the same ---
    # Use 'selected_task_config' instead of the old single 'task_config'
    # Generate prompt templates
    prompts = generate_prompt_templates(initial_prompt)
    print("All prompt templates generated.")

    print("Loading model list...")
    if ENGINE_CONFIG.get('CUSTOM_MODELS', {}).get('HF_MODELS') and not ENGINE_CONFIG.get('CUSTOM_MODELS', {}).get('HF_API_TOKEN'):
        print("Warning: HF_MODELS are specified, but HF_API_TOKEN is not set. These models will likely fail.", file=sys.stderr)

>>>>>>> 378884a05e1a0098f1c86c0a0cb10d3b3c2af67e
    try:
        models = get_models_list(ENGINE_CONFIG)
        if not models:
            print("No models found. Exiting.", file=sys.stderr)
            sys.exit(1)
    except Exception as e:
        print(f"Failed to get model list: {e}", file=sys.stderr)
        sys.exit(1)

<<<<<<< HEAD
=======
    intermediate_folder = ENGINE_CONFIG['CONSTANTS']['INTERMEDIATE_FOLDER']
    if not os.path.exists(intermediate_folder):
        try:
            os.makedirs(intermediate_folder)
        except OSError as e:
            print(f"Failed to create folder {intermediate_folder}: {e}", file=sys.stderr)
            return

    all_results = {}
    MAX_MODELS_TO_TEST = -1 # -1 for all
>>>>>>> 378884a05e1a0098f1c86c0a0cb10d3b3c2af67e
    if MAX_MODELS_TO_TEST > 0:
        models = models[:MAX_MODELS_TO_TEST]

<<<<<<< HEAD
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
=======
    try:
        # Pass the selected_task_config (which contains test_code_func and TASK_CONSTANTS)
        # to the process_model function.
        with ThreadPoolExecutor(max_workers=ENGINE_CONFIG['CONSTANTS']['MAX_WORKERS']) as executor:
            futures = {
                executor.submit(process_model, model, selected_task_config, prompts, ENGINE_CONFIG, progress_queue): model
                for model in models_to_test
            }
            # ... (rest of the executor logic remains the same)
            completed_count = 0
            total_count = len(futures)
            start_time_main = perf_counter()
            while completed_count < total_count:
                done_futures = [f for f in futures if f.done()]
                for future in done_futures:
                    model = futures.pop(future)
                    completed_count += 1
                    try:
                        result = future.result()
                        all_results[model] = result
                        final_success = result.get('final_test', {}).get('success', False)
                        status_str = "SUCCESS" if final_success else "FAILED"
                        if result.get('final_code'):
                            code_filename = f"{model.replace('/', '_')}_final.py"
                            code_path = os.path.join(intermediate_folder, code_filename)
                            try:
                                with open(code_path, 'w', encoding='utf-8') as f:
                                    f.write(result['final_code'])
                            except Exception as e:
                                print(f"Error saving code for {model}: {e}", file=sys.stderr)
                        print(f"--- ({completed_count}/{total_count}) COMPLETED: {model} [Status: {status_str}] ---")
                    except Exception as e:
                        print(f"--- ({completed_count+1}/{total_count}) CRITICAL ERROR (Executor): {model} ---")
                        tb_str = traceback.format_exc()
                        print(tb_str, file=sys.stderr)
                        all_results[model] = {'error': str(e), 'traceback': tb_str, 'iterations': [], 'final_code': None, 'final_test': {'success': False, 'summary': None, 'issue': str(e)}}
                    if completed_count % ENGINE_CONFIG['CONSTANTS']['N_SAVE'] == 0 or completed_count == total_count:
                        save_results(all_results, intermediate_folder, f"intermediate_results_{completed_count}.json")
                try:
                    while not progress_queue.empty():
                        model, type, message = progress_queue.get_nowait()
                        # (Uncomment for detailed logs)
                        # if type == 'log':
                        #     print(f"LOG [{model}]: {message}")
                        # elif type == 'status':
                        #     print(f"STATUS [{model}]: {message}")
                except queue.Empty:
                    pass
                time.sleep(0.2)
    except KeyboardInterrupt:
        print("\nStopping at user request... (Waiting for current threads to finish)")
    finally:
        end_time_main = perf_counter()
        total_time_main = end_time_main - start_time_main
        print("--- TESTING FINISHED ---")
        print(f"Total execution time: {total_time_main:.2f} sec.")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        final_filename = f"final_results_{timestamp}.json"
        save_results(all_results, intermediate_folder, final_filename)
        print(f"Final results saved to: {os.path.join(intermediate_folder, final_filename)}")
        success_count = sum(1 for res in all_results.values() if res.get('final_test', {}).get('success', False))
        fail_count = len(all_results) - success_count
        print(f"Totals: {success_count} Successful, {fail_count} Failed.")
>>>>>>> 378884a05e1a0098f1c86c0a0cb10d3b3c2af67e

if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
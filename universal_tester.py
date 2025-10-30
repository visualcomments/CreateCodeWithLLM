import requests
import json
import os
import sys
import importlib.util
from concurrent.futures import ThreadPoolExecutor, as_completed
import subprocess
from typing import Dict, List, Optional, Tuple, Any
import time
from datetime import datetime
import random
import psutil
from time import perf_counter
import re
import tempfile
import traceback 

# Patch for RotatedProvider (used in AnyProvider for rotation)
try:
    import g4f.providers.retry_provider as retry_mod  # Import module
    OriginalRotatedProvider = retry_mod.RotatedProvider  # Alias original for inheritance
except ImportError:
    print("Failed to import g4f.providers.retry_provider. Using fallback.", file=sys.stderr)
    # Fallback if g4f is not installed or structure changed
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
        
        # Check if self.providers is empty (can happen with g4f errors)
        if not self.providers:
             raise ModelNotFoundError(f"No providers found for model {model}", [])

        for provider_class in self.providers:
            p = None
            # Safely get provider name BEFORE try (for str/classes)
            if isinstance(provider_class, str):
                provider_name = provider_class
            else:
                provider_name = provider_class.__name__ if hasattr(provider_class, '__name__') else str(provider_class)
            current_data['tried'].append(provider_name)
            if hasattr(local, 'current_model') and hasattr(local, 'current_queue'):
                local.current_queue.put((local.current_model, 'log', f'2) Trying {provider_name} with model: {model}'))
            try:
                # If str, convert to class for instantiation
                if isinstance(provider_class, str):
                    if hasattr(Provider, provider_class):
                        provider_class = getattr(Provider, provider_class)
                    else:
                        raise ValueError(f"Provider '{provider_name}' not found in Provider")
                p = provider_class()
                async for chunk in p.create_async_generator(model, messages, **kwargs):
                    yield chunk
                # Success: put log
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
        # No success: final log
        try:
            if hasattr(local, 'current_model') and hasattr(local, 'current_queue'):
                local.current_queue.put((local.current_model, 'log', f'Debug: TrackedRotated finished, tried_providers={current_data["tried"]}'))
        except Exception:
            pass
        raise ModelNotFoundError(f"No working provider for model {model}", current_data['tried'])

# Monkey-patch: replace RotatedProvider with TrackedRotated (used by AnyProvider)
try:
    retry_mod.RotatedProvider = TrackedRotated
except NameError:
    print("Failed to apply Monkey-patch for RotatedProvider (retry_mod not defined)", file=sys.stderr)


# Patch g4f.debug to write to queue (no console, with JSON if needed)
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
# ... [End of Patches] ...


# =============================================================================
# === ENGINE CONFIGURATION ===
# =============================================================================

ENGINE_CONFIG = {
    'URLS': {
        'WORKING_RESULTS': '[https://raw.githubusercontent.com/maruf009sultan/g4f-working/refs/heads/main/working/working_results.txt](https://raw.githubusercontent.com/maruf009sultan/g4f-working/refs/heads/main/working/working_results.txt)'
    },
    'RETRIES': {
        'INITIAL': {'max_retries': 1, 'backoff_factor': 1.0},
        'FIX': {'max_retries': 3, 'backoff_factor': 2.0},
        'HARNESS_GEN': {'max_retries': 3, 'backoff_factor': 2.0}, # For the test generator
        'HARNESS_TEST': {'max_retries': 1, 'backoff_factor': 1.0}  # For testing the generator models
    },
    'CUSTOM_MODELS': {
        'HF_MODELS': [
            # "hf:meta-llama/Llama-2-7b-chat-hf",
        ],
        'HF_API_URL': os.environ.get("HF_API_URL", None),
        'HF_API_TOKEN': os.environ.get("HF_API_TOKEN", None)
    },
    'CONSTANTS': {
        'DELIMITER_MODEL': '|',
        'MODEL_TYPE_TEXT': 'text',
        'REQUEST_TIMEOUT': 10,
        'N_SAVE': 100,
        'MAX_WORKERS': 10,
        'ERROR_NO_RESPONSE': 'No response from model',
        'NUM_REFACTOR_LOOPS': 3,
        'INTERMEDIATE_FOLDER': 'results',
        
        # --- NEW: List of models to try for harness generation ---
        # It will test these in order and use the first one that works.
        'HARNESS_GENERATOR_MODELS': [
            'deepseek-v3',                 # User's requested model
            'gpt-4o-mini',                 # A strong, reliable choice
            g4f.models.gemini_pro,         # A good fallback
            g4f.models.claude_3_haiku,     # Another strong option
            g4f.models.claude_3_sonnet
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

# =============================================================================
# === PROMPT TEMPLATES ===
# =============================================================================

# Meta-prompt to generate the test_code function
META_PROMPT_TEMPLATE = r"""
You are an expert Test Driven Development (TDD) engineer. Your task is to generate a Python script containing a test harness for a given algorithmic task description.

The generated script MUST contain:
1.  A dictionary named `TASK_CONSTANTS`. It must include `EXEC_TIMEOUT` (in seconds, estimate a reasonable value based on the task description, e.g., 10s for simple poly-time, 30s+ for search/BFS on small inputs) and `ERROR_TIMEOUT` (a string message).
2.  A function `test_code(code: str, task_config: Dict) -> Tuple[bool, str, Optional[Dict]]`. This function MUST be self-contained *except* for standard libraries (json, sys, subprocess, tempfile, os, psutil, traceback, time, typing, random, queue, collections).

The `test_code` function must:
-   Accept the Python `code` as a string and `task_config` (which will be the `TASK_CONSTANTS` dict).
-   Define "ground truth" logic based *only* on the task description (e.g., if the task is 'L, R, X' sorting, it must define `_apply_move` functions).
-   Define a list of test cases (e.g., `specific_vectors`) including edge cases (empty, single, already sorted, reverse, etc.).
-   Write the `code` string to a temporary file.
-   Run the temporary file as a subprocess (`sys.executable`) for each test case, passing the test input as a JSON string in `sys.argv[1]`.
-   Use `subprocess.communicate` with the `EXEC_TIMEOUT` from `task_config`.
-   Parse the subprocess `stdout` as JSON.
-   Verify the JSON output structure (e.g., keys "moves", "sorted_array").
-   Verify the *correctness* of the output using the ground-truth logic (e.g., check if the array is sorted AND if applying the "moves" list to the input actually produces the sorted array).
-   Return `(True, "All tests passed", summary_dict)` on success.
-   Return `(False, "Error message", summary_dict)` on failure (timeout, JSON error, logic error, etc.).

TASK DESCRIPTION:
---
{task_prompt}
---

Your response MUST be *only* the raw, executable Python code containing `TASK_CONSTANTS` and `test_code`. Do not include explanations, markdowns, or any other text.
"""

# Template for FIX (automatically inserts the full task prompt)
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

# Template for REFACTOR (automatically inserts the full task prompt)
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

# Template for REFACTOR_NO_PREV (automatically inserts the full task prompt)
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

def get_models_list(config: Dict) -> List[str]:
    """
    Forms a list of available models from g4f and custom config.
    """
    working_models = set()
    url_txt = config['URLS']['WORKING_RESULTS']
    try:
        resp = requests.get(url_txt, timeout=config['CONSTANTS']['REQUEST_TIMEOUT'])
        resp.raise_for_status()
        text = resp.text
        for line in text.splitlines():
            if config['CONSTANTS']['DELIMITER_MODEL'] in line:
                parts = [p.strip() for p in line.split(config['CONSTANTS']['DELIMITER_MODEL'])]
                if len(parts) == 3 and parts[2] == config['CONSTANTS']['MODEL_TYPE_TEXT']:
                    model_name = parts[1]
                    if 'flux' not in model_name.lower():
                        working_models.add(model_name)
    except requests.RequestException as e:
        print(f"Warning: Failed to download {url_txt}. Reason: {e}. Using only g4f.models.", file=sys.stderr)
        text = ''
    
    try:
        from g4f.models import Model
        all_g4f_models = Model.__all__()
        g4f_models = set()
        for model_name in all_g4f_models:
            if 'flux' not in model_name.lower() and not any(sub in model_name.lower() for sub in ['image', 'vision', 'audio', 'video']):
                g4f_models.add(model_name)
    except ImportError:
        print("Warning: Failed to import g4f.models. Model list may be incomplete.", file=sys.stderr)
        g4f_models = set()
    
    all_models_set = working_models.union(g4f_models)

    if 'CUSTOM_MODELS' in config and 'HF_MODELS' in config['CUSTOM_MODELS']:
        hf_models = config['CUSTOM_MODELS']['HF_MODELS']
        if hf_models:
            print(f"Adding {len(hf_models)} custom HuggingFace models...")
            all_models_set.update(hf_models)
    
    all_models = list(all_models_set)
    all_models = [m for m in all_models if m not in ['sldx-turbo', 'turbo']]
    return all_models


def llm_query(model: Any, prompt: str, retries_config: Dict, config: Dict, progress_queue: queue.Queue, stage: str = None) -> Optional[str]:
    """
    Query LLM, now supporting g4f Model objects, string names, and 'hf:' prefixed strings.
    """
    # Handle g4f Model objects by getting their name
    if isinstance(model, g4f.models.Model):
        model_name_str = model.name
    else:
        model_name_str = str(model)

    local.current_model = model_name_str
    local.current_queue = progress_queue
    local.current_data = {'tried': [], 'errors': {}, 'success': None, 'model': model_name_str}
    local.current_stage = stage

    request_timeout = config['CONSTANTS']['REQUEST_TIMEOUT']
    
    is_hf_model = model_name_str.startswith('hf:')

    for attempt in range(retries_config['max_retries'] + 1):
        try:
            if is_hf_model:
                hf_model_name = model_name_str.split(':', 1)[1]
                token = config['CUSTOM_MODELS']['HF_API_TOKEN']
                host = config['CUSTOM_MODELS']['HF_API_URL']

                if not token:
                    progress_queue.put((model_name_str, 'log', 'Error: HF_API_TOKEN not set in env or config. Skipping HuggingFace model.'))
                    return None

                progress_queue.put((model_name_str, 'log', f'Using HuggingFace provider for {hf_model_name} (Attempt {attempt+1})'))
                
                response = g4f.ChatCompletion.create(
                    model=hf_model_name,
                    messages=[{"role": "user", "content": prompt}],
                    provider=Provider.HuggingFace,
                    auth=token,
                    api_host=host,
                    timeout=request_timeout * 2
                )
            
            else:
                # Use the 'model' argument directly, which can be an object or string
                response = g4f.ChatCompletion.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    provider=Provider.AnyProvider,
                    timeout=request_timeout
                )
            
            if response and response.strip():
                if not is_hf_model:
                    s_provider = local.current_data.get('success', 'Unknown')
                    progress_queue.put((model_name_str, 'log', f'Success from provider: {s_provider}'))
                else:
                    progress_queue.put((model_name_str, 'log', f'Success from HuggingFace: {hf_model_name}'))
                return response.strip()

        except ModelNotFoundError as e:
            progress_queue.put((model_name_str, 'log', f'Error: ModelNotFoundError: {e}'))
            if len(e.args) > 1:
                local.current_data['tried'] = e.args[1]
            return None
        except Exception as e:
            error_msg = f'g4f Error (attempt {attempt+1}): {e}'
            progress_queue.put((model_name_str, 'log', error_msg))
            if is_hf_model:
                local.current_data['errors']['HuggingFace'] = str(e)
            
        
        if attempt < retries_config['max_retries']:
            time.sleep(retries_config['backoff_factor'] * (2 ** attempt))

    return None

def find_working_harness_model(config: Dict, progress_queue: queue.Queue) -> Optional[Any]:
    """
    Tests models from HARNESS_GENERATOR_MODELS list and returns the first one that works.
    """
    print("--- Finding a working Harness Generator Model ---")
    test_prompt = "Respond with only the single word: 'OK'"
    models_to_test = config['CONSTANTS']['HARNESS_GENERATOR_MODELS']
    
    if not models_to_test:
        print("Error: `HARNESS_GENERATOR_MODELS` list in config is empty.", file=sys.stderr)
        return None
        
    for model in models_to_test:
        model_name = model.name if isinstance(model, g4f.models.Model) else str(model)
        print(f"Testing model: {model_name}...")
        progress_queue.put(("HARNESS_TEST", 'log', f"Pinging model: {model_name}"))
        
        response = llm_query(
            model=model,
            prompt=test_prompt,
            retries_config=config['RETRIES']['HARNESS_TEST'],
            config=config,
            progress_queue=progress_queue,
            stage='harness_test'
        )
        
        if response and 'ok' in response.lower().strip():
            print(f"SUCCESS: Model {model_name} is working and will be used.")
            progress_queue.put(("HARNESS_TEST", 'log', f"SUCCESS: Selected {model_name}"))
            return model
        else:
            print(f"FAILED: Model {model_name} did not respond correctly.")
            progress_queue.put(("HARNESS_TEST", 'log', f"FAILED: {model_name} did not respond 'OK'"))

    print("--- CRITICAL: No working harness generator model found. ---", file=sys.stderr)
    return None

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
    
    meta_prompt = META_PROMPT_TEMPLATE.format(task_prompt=initial_prompt)
    model_name_str = harness_model.name if isinstance(harness_model, g4f.models.Model) else str(harness_model)
    
    print(f"Using validated model: {model_name_str} to generate test_code function...")
    progress_queue.put(("HARNESS_GEN", 'log', f"Calling {model_name_str} with meta-prompt..."))

    response_code = llm_query(
        model=harness_model,
        prompt=meta_prompt,
        retries_config=engine_config['RETRIES']['HARNESS_GEN'],
        config=engine_config,
        progress_queue=progress_queue,
        stage='generate_harness'
    )

    if not response_code:
        print("FATAL ERROR: LLM failed to generate test harness code.", file=sys.stderr)
        progress_queue.put(("HARNESS_GEN", 'log', "FATAL: No response for meta-prompt."))
        return None

    cleaned_code = clean_code(response_code)
    
    if not cleaned_code:
        print("FATAL ERROR: LLM response for harness was empty after cleaning.", file=sys.stderr)
        print(f"Original response: {response_code}", file=sys.stderr)
        return None

    print("Test harness code received. Executing to load definitions...")
    
    # --- Dynamic execution of the LLM-generated code ---
    context = {}
    try:
        # Provide necessary imports for the generated code to run
        exec(cleaned_code, {
            'json': json, 'sys': sys, 'subprocess': subprocess, 'tempfile': tempfile,
            'os': os, 'psutil': psutil, 'traceback': traceback, 'time': time,
            'perf_counter': perf_counter, 'Dict': Dict, 'List': List, 
            'Optional': Optional, 'Tuple': Tuple, 'random': random,
            'queue': queue,
            # Add other common libraries the test function might need
            're': re, 
            'collections': __import__('collections')
        }, context)
    except Exception as e:
        print(f"FATAL ERROR: Failed to execute generated test harness code.", file=sys.stderr)
        print(f"Error: {e}", file=sys.stderr)
        print(f"--- Generated Code ---:\n{cleaned_code}\n---", file=sys.stderr)
        traceback.print_exc()
        return None

    if 'test_code' not in context or not callable(context['test_code']):
        print(f"FATAL ERROR: Generated code did not define a callable `test_code` function.", file=sys.stderr)
        return None
        
    if 'TASK_CONSTANTS' not in context or not isinstance(context['TASK_CONSTANTS'], dict):
        print(f"FATAL ERROR: Generated code did not define a `TASK_CONSTANTS` dictionary.", file=sys.stderr)
        return None

    print("--- Task Harness Generation SUCCESS ---")
    
    return {
        "test_code_func": context['test_code'],
        "TASK_CONSTANTS": context['TASK_CONSTANTS']
    }


def process_model(model: str, task_config: Dict, prompts: Dict, engine_config: Dict, progress_queue: queue.Queue) -> Dict:
    """
    Process one model: sequence of LLM queries, test, fix, refactor.
    """
    iterations = []
    current_code = None
    prev_code = None
    early_stop = False
    
    TASK_CONSTANTS = task_config['TASK_CONSTANTS']
    test_code_func = task_config['test_code_func']
    
    STAGES = engine_config['STAGES']
    RETRIES = engine_config['RETRIES']
    CONSTANTS = engine_config['CONSTANTS']

    num_loops = CONSTANTS['NUM_REFACTOR_LOOPS']
    total_stages = 1 + 1 + 1 + 1 + (num_loops * 2) + 1
    current_stage_count = 0

    def update_progress(stage_name):
        nonlocal current_stage_count
        current_stage_count += 1
        progress_queue.put((model, 'status', f'Stage: {stage_name} ({current_stage_count}/{total_stages})'))
        progress_queue.put((model, 'progress', (current_stage_count, total_stages)))

    def run_test(code_to_test, stage_name):
        """Internal function for testing. Calls dynamically generated test_code_func."""
        if not code_to_test or not code_to_test.strip():
            progress_queue.put((model, 'log', f'Test {stage_name}: Skipped (no code).'))
            return False, "No code to test", None
            
        progress_queue.put((model, 'log', f'Test {stage_name}: Running generated test_code_func...'))
        
        try:
            success, issue, summary = test_code_func(code_to_test, TASK_CONSTANTS)
        except Exception as e:
            tb_str = traceback.format_exc()
            print(f"CRITICAL ERROR in generated test_code_func for model {model}:\n{tb_str}", file=sys.stderr)
            success = False
            issue = f"Error during generated test_code execution: {e}"
            summary = {'error': issue, 'traceback': tb_str}
        
        if success:
            progress_queue.put((model, 'log', f'Test {stage_name}: SUCCESS. {issue}'))
        else:
            progress_queue.put((model, 'log', f'Test {stage_name}: FAILED. Reason: {issue}'))
        return success, issue, summary

    def run_llm_query(prompt, stage_name, retries_key='FIX'):
        """Internal function for LLM query. Passes engine_config."""
        progress_queue.put((model, 'log', f'Stage: {stage_name}. Prompt:\n{prompt[:500]}...'))
        retries_cfg = RETRIES[retries_key]
        progress_queue.put((model, 'log', f'Calling llm_query with retries: {retries_cfg}'))
        
        response = llm_query(model, prompt, retries_cfg, engine_config, progress_queue, stage_name)
        
        tried = local.current_data.get('tried', [])
        success_p = local.current_data.get('success', None)
        
        if response:
            cleaned = clean_code(response)
            progress_queue.put((model, 'log', f'Received response (length: {len(response)}), cleaned (length: {len(cleaned)}):\n{cleaned[:500]}...'))
            return cleaned, None, tried, success_p
        else:
            error_msg = CONSTANTS['ERROR_NO_RESPONSE']
            progress_queue.put((model, 'log', f'llm_query error: {error_msg}'))
            return None, error_msg, tried, success_p

    def add_iteration(stage, response, error, test_summary, tried, success_p):
        iterations.append({
            'providers_tried': tried,
            'success_provider': success_p,
            'stage': stage,
            'response': response,
            'error': error,
            'test_summary': test_summary
        })

    # ---
    # START OF PROCESS
    # ---
    progress_queue.put((model, 'log', f'=== STARTING MODEL PROCESSING: {model} ==='))
    
    # 1. Initial
    stage = STAGES['INITIAL']
    update_progress(stage)
    prompt = prompts['INITIAL']
    current_code, llm_error, tried, s_provider = run_llm_query(prompt, stage, 'INITIAL')
    add_iteration(stage, current_code, llm_error, None, tried, s_provider)
    if llm_error:
        progress_queue.put((model, 'status', f'Error at stage: {stage}'))
        return {'model': model, 'iterations': iterations, 'final_code': None,
                'final_test': {'success': False, 'summary': None, 'issue': 'No initial response'}}

    # 2. Test & Fix (Initial)
    stage = STAGES['FIX_INITIAL']
    update_progress(stage)
    success, issue, summary = run_test(current_code, stage)
    if not success:
        prompt = prompts['FIX'].format(code=current_code, error=issue)
        current_code, llm_error, tried, s_provider = run_llm_query(prompt, stage)
        add_iteration(stage, current_code, llm_error, summary, tried, s_provider)
        if llm_error:
            early_stop = True
    else:
        add_iteration(stage, current_code, None, summary, [], None) 
    if early_stop:
        progress_queue.put((model, 'status', f'Error at stage: {stage}'))
        return {'model': model, 'iterations': iterations, 'final_code': current_code,
                'final_test': {'success': False, 'summary': summary, 'issue': f'LLM error during {stage}'}}

    # 3. Refactor (First)
    prev_code = current_code
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
            early_stop = True
    else:
        add_iteration(stage, current_code, None, summary, [], None)
    if early_stop:
        progress_queue.put((model, 'status', f'Error at stage: {stage}'))
        return {'model': model, 'iterations': iterations, 'final_code': current_code,
                'final_test': {'success': False, 'summary': summary, 'issue': f'LLM error during {stage}'}}

    # 5. Refactor Loops (N times)
    for i in range(CONSTANTS['NUM_REFACTOR_LOOPS']):
        if not current_code or not current_code.strip():
            progress_queue.put((model, 'log', f'Skipping refactor loop {i+1} (no code).'))
            update_progress(f'loop {i+1} refactor (skip)')
            update_progress(f'loop {i+1} fix (skip)')
            continue
        
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
                break
        else:
            add_iteration(stage, current_code, None, summary, [], None)

    # 6. Final Test
    stage = 'final_test'
    update_progress(stage)
    success, issue, summary = run_test(current_code, stage)
    add_iteration(stage, current_code, None if success else issue, summary, [], None)
    if success:
        progress_queue.put((model, 'log', f'FINAL: SUCCESS. {issue}'))
        progress_queue.put((model, 'status', 'Success (final test)'))
    else:
        progress_queue.put((model, 'log', f'FINAL: FAILED. Reason: {issue}'))
        progress_queue.put((model, 'status', 'Failed (final test)'))

    return {
        'model': model,
        'iterations': iterations,
        'final_code': current_code,
        'final_test': {'success': success, 'summary': summary, 'issue': issue}
    }


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

def main():
    """
    Main function: Loads task prompt from .txt, generates harness, 
    loads models, and starts thread pool.
    """
    
    if len(sys.argv) < 2:
        print("Usage: python universal_tester_v2_en.py <path_to_task_prompt.txt>", file=sys.stderr)
        sys.exit(1)
    
    task_prompt_path = sys.argv[1]
    try:
        with open(task_prompt_path, 'r', encoding='utf-8') as f:
            initial_prompt = f.read()
        if not initial_prompt.strip():
            print(f"Error: Task prompt file is empty: {task_prompt_path}", file=sys.stderr)
            sys.exit(1)
        print(f"Loaded task prompt from: {task_prompt_path}")
    except FileNotFoundError:
        print(f"Error: Task prompt file not found: {task_prompt_path}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error reading task prompt file: {e}", file=sys.stderr)
        sys.exit(1)

    # --- NEW: Generate all components dynamically ---
    progress_queue = queue.Queue()

    # 1. Find a working model for harness generation
    working_harness_model = find_working_harness_model(ENGINE_CONFIG, progress_queue)
    if working_harness_model is None:
        print("FATAL: Could not find a working harness generator model. Exiting.", file=sys.stderr)
        while not progress_queue.empty():
            print(f"LOG: {progress_queue.get_nowait()}", file=sys.stderr)
        sys.exit(1)

    # 2. Generate task harness (test_code, TASK_CONSTANTS)
    task_config = generate_task_harness(
        initial_prompt, 
        working_harness_model, 
        ENGINE_CONFIG, 
        progress_queue
    )
    if task_config is None:
        print("FATAL: Could not generate task harness. Exiting.", file=sys.stderr)
        while not progress_queue.empty():
            print(f"LOG: {progress_queue.get_nowait()}", file=sys.stderr)
        sys.exit(1)
    
    # Save the generated test harness for inspection
    try:
        generated_harness_path = os.path.join(
            ENGINE_CONFIG['CONSTANTS']['INTERMEDIATE_FOLDER'], 
            "__generated_test_harness.py"
        )
        if 'test_code_func' in task_config and hasattr(task_config['test_code_func'], '__code__'):
            import inspect
            harness_code = inspect.getsource(task_config['test_code_func'])
            constants_code = f"TASK_CONSTANTS = {json.dumps(task_config['TASK_CONSTANTS'], indent=4)}\n\n"
            
            # We need to find the imports from the exec context
            # This is complex; for now, just save the function and constants
            save_results({}, ENGINE_CONFIG['CONSTANTS']['INTERMEDIATE_FOLDER'], "dummy_init_folder.json") # Ensure folder exists
            
            with open(generated_harness_path, 'w', encoding='utf-8') as f:
                f.write("# --- Auto-generated by universal_tester_v2_en.py ---\n")
                f.write("# This file contains the dynamically generated test harness for inspection.\n")
                f.write(f"# Generated at: {datetime.now().isoformat()}\n\n")
                f.write("# Required imports (must be manually added if running this file standalone):\n")
                f.write("# import json, sys, subprocess, tempfile, os, psutil, traceback, time, re, collections, queue\n")
                f.write("# from typing import Dict, List, Optional, Tuple\n")
                f.write("# from time import perf_counter\n\n")
                f.write(constants_code)
                f.write(harness_code)
            print(f"Saved generated test harness for inspection to: {generated_harness_path}")
    except Exception as e:
        print(f"Warning: Could not save generated test harness for inspection. Error: {e}", file=sys.stderr)


    # 3. Generate prompt templates
    prompts = generate_prompt_templates(initial_prompt)
    print("All prompt templates generated.")
    
    # --- End of new dynamic generation ---

    print("Loading model list...")
    if ENGINE_CONFIG.get('CUSTOM_MODELS', {}).get('HF_MODELS') and not ENGINE_CONFIG.get('CUSTOM_MODELS', {}).get('HF_API_TOKEN'):
        print("Warning: HF_MODELS are specified, but HF_API_TOKEN is not set. These models will likely fail.", file=sys.stderr)
        
    try:
        models = get_models_list(ENGINE_CONFIG)
        if not models:
            print("No models found. Check URLS, g4f.models, and CUSTOM_MODELS.", file=sys.stderr)
            return
        print(f"Found {len(models)} unique models for testing.")
    except Exception as e:
        print(f"Failed to load model list: {e}", file=sys.stderr)
        traceback.print_exc()
        return

    intermediate_folder = ENGINE_CONFIG['CONSTANTS']['INTERMEDIATE_FOLDER']
    if not os.path.exists(intermediate_folder):
        try:
            os.makedirs(intermediate_folder)
        except OSError as e:
            print(f"Failed to create folder {intermediate_folder}: {e}", file=sys.stderr)
            return

    all_results = {}
    MAX_MODELS_TO_TEST = -1 # -1 for all
    
    if MAX_MODELS_TO_TEST > 0:
        models_to_test = models[:MAX_MODELS_TO_TEST]
        print(f"--- STARTING TEST (Limited to {len(models_to_test)} models) ---")
    else:
        models_to_test = models
        print(f"--- STARTING TEST (All {len(models_to_test)} models) ---")


    try:
        with ThreadPoolExecutor(max_workers=ENGINE_CONFIG['CONSTANTS']['MAX_WORKERS']) as executor:
            futures = {
                executor.submit(process_model, model, task_config, prompts, ENGINE_CONFIG, progress_queue): model 
                for model in models_to_test
            }
            
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


if __name__ == "__main__":
    main()
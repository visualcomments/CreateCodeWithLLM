import requests
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import subprocess
import sys
from typing import Dict, List, Optional, Tuple
import time
from datetime import datetime
import random
import psutil
from time import perf_counter
import re
import tempfile
import traceback # <-- Improvement: for detailed error logging

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
    if content_from_json is None or not content_from_json.strip():
        # Find first block ```python\n... (up to next ``` or end)
        match = re.search(r'```(?:python)?\s*\n(.*?)(?=\n?```\s*$|\Z)', code, re.DOTALL | re.MULTILINE)
        if match:
            code = match.group(1)
        # Alternative: if block without closing ```, search from first ``` to end
        else:
            match = re.search(r'```(?:python)?\s*\n(.*)', code, re.DOTALL | re.MULTILINE)
            if match:
                code = match.group(1)

    # Step 3: Final regex cleanup (for nested markdown)
    # Remove ```python block at start
    code = re.sub(r'^```(?:python)?\s*\n?', '', code, flags=re.MULTILINE)
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


CONFIG = {
    # Section with URLs for downloading data about working models
    'URLS': {
        # URL of the file with test results for working g4f models
        # *** FIX: Removed Markdown formatting from the URL ***
        'WORKING_RESULTS': '[https://raw.githubusercontent.com/maruf009sultan/g4f-working/refs/heads/main/working/working_results.txt](https://raw.githubusercontent.com/maruf009sultan/g4f-working/refs/heads/main/working/working_results.txt)'
    },

    # Section with prompts for various stages of interaction with LLM
    'PROMPTS': {
        # This prompt contains {n-1}, which breaks .format(task=...).
        # We will no longer format it.
        'INITIAL': r"""
You are an AI assistant. Your task is to write code that implements the three fundamental transformations of the "LRX algorithm". These transformations operate on a permutation of $n$ elements.

Based on the definitions:
L (Left Cyclic Shift): [e_1, e_2, ..., e_n] becomes [e_2, ..., e_n, e_1]
R (Right Cyclic Shift): [e_1, e_2, ..., e_n] becomes [e_n, e_1, ..., e_{n-1}]
X (Transposition): [e_1, e_2, ..., e_n] becomes [e_2, e_1, ..., e_n]

⚠️ Requirements:
• Implement three distinct functions `L(vector)`, `R(vector)`, and `X(vector)`.
• Each function must accept a list/array and return the **new** list/array. The original list must not be modified.
• Include `import json` and a CLI entry point: when executed, parse `sys.argv[1]` as JSON vector, fallback [1, 2, 3, 4] if missing or invalid.
• In the `__main__` block, call L, R, and X on the input vector.
• Print only one JSON object: `{{"L_result": L(vector), "R_result": R(vector), "X_result": X(vector)}}`.
• The JSON output must be structured and parseable (double quotes for keys).
• Fully self-contained and immediately executable.
• Only code in the response, no explanations or Markdown.
""",

        'FIX': r"""
You are a Python debugging assistant. The following code did not work correctly.
Fix it to run correctly, follow the `L, R, X` function interfaces, and produce **only CLI JSON output**.

Code:
{code}

Issue:
{error}

⚠️ Requirements:
• `L(vector)`, `R(vector)`, `X(vector)` must return **new** lists, not modify the input.
• CLI: `import json`; parse `sys.argv[1]` with fallback [1, 2, 3, 4]; print only `json.dumps({{"L_result": L(vector), "R_result": R(vector), "X_result": X(vector)}})`.
• Use try-except to catch missing arguments or invalid JSON.
• Self-contained, executable, immediately correct.
• Only code in response, no extra prints or Markdown.
""",

        'REFACTOR_NO_PREV': r"""
You are an expert Python programmer. Refactor the following code:

{code}

⚠️ Goals:
• Improve readability, structure, and efficiency of `L`, `R`, `X` (e.g., using slicing for new lists).
• Ensure functions return **new** lists and do not modify the input.
• Preserve CLI: parse `sys.argv[1]` as JSON with fallback [1, 2, 3, 4], print only `json.dumps({{"L_result": L(vector), "R_result": R(vector), "X_result": X(vector)}})`.
• Minimal example in __main__ must print JSON only.
• Fully executable, immediately correct.
• Only code in response, no explanations or Markdown.
""",

        'REFACTOR': r"""
You are an expert Python programmer. Compare the current and previous versions and perform a full refactor:

Current code:
{code}

Previous version:
{prev}

⚠️ Goals:
• Improve readability, structure, and efficiency of `L`, `R`, `X`.
• Ensure functions return **new** lists and do not modify the input.
• Preserve CLI: parse `sys.argv[1]` as JSON with fallback [1, 2, 3, 4]; print only `json.dumps({{"L_result": L(vector), "R_result": R(vector), "X_result": X(vector)}})`.
• Minimal example in __main__ must print JSON only.
• Code must pass verification (e.g., L([1,2,3]) == [2,3,1], R([1,2,3]) == [3,1,2], X([1,2,3]) == [2,1,3]).
• Only code in response, no explanations or Markdown.
"""
    },

    # Section with retry settings for different request types
    'RETRIES': {
        # Retry settings for initial request: max 1 retry, backoff factor 1.0
        'INITIAL': {'max_retries': 1, 'backoff_factor': 1.0},
        # Retry settings for fixes: max 3 retries, backoff factor 2.0 (exponential)
        'FIX': {'max_retries': 3, 'backoff_factor': 2.0}
    },

    # Section with system constants
    'CONSTANTS': {
        # Delimiter in working_results.txt file strings (between Provider|Model|Type)
        'DELIMITER_MODEL': '|',
        # Model type for filtering (only text models)
        'MODEL_TYPE_TEXT': 'text',
        # Timeout for URL requests (in seconds)
        'REQUEST_TIMEOUT': 10, # Reduced for faster fallback
        # Frequency for saving intermediate results (every N models)
        'N_SAVE': 100,
        # Maximum number of parallel threads for processing models
        'MAX_WORKERS': 10, # IMPROVEMENT: 50 is too much for Kaggle
        # Timeout for code execution in subprocess (in seconds)
        'EXEC_TIMEOUT': 5,
        # Error message for code execution timeout
        'ERROR_TIMEOUT': 'Timeout expired — the program likely entered an infinite loop.',
        # Error message for no response from model
        'ERROR_NO_RESPONSE': 'No response from model',
        # Number of refactoring loops in process_model
        'NUM_REFACTOR_LOOPS': 3,
        # Folder name for intermediate and final results
        # *** FIX: Translated to English ***
        'INTERMEDIATE_FOLDER': 'results'
    },

    # Section with stage names for logs and statuses
    # *** FIX: Translated to English ***
    'STAGES': {
        # Stage for generating initial code
        'INITIAL': 'initial_response',
        # Stage for fixing code before first refactor
        'FIX_INITIAL': 'fix_before_refactor',
        # Stage for first refactor
        'REFACTOR_FIRST': 'refactor_first_response',
        # Stage for fixing after first refactor
        'FIX_AFTER_REFACTOR': 'fix_after_refactor',
        # Stage for refactoring in a loop
        'REFACTOR': 'refactor_loop',
        # Stage for fixing in a refactor loop
        'FIX_LOOP': 'fix_loop'
    }
}

def get_models_list(config: Dict) -> List[str]:
    """
    Function to form a list of available models.
    IMPROVEMENT: Added network error handling.
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
                    # Additional filter: exclude flux and similar
                    if 'flux' not in model_name.lower():
                        working_models.add(model_name)
    except requests.RequestException as e:
        # *** FIX: Translated message ***
        print(f"Warning: Failed to download {url_txt}. Reason: {e}. Using only g4f.models.", file=sys.stderr)
        text = ''
    
    # From g4f.models: only base text Models, excluding subclasses (Image, Vision, etc.)
    try:
        from g4f.models import Model
        all_g4f_models = Model.__all__()
        g4f_models = set()
        for model_name in all_g4f_models:
            if 'flux' not in model_name.lower() and not any(sub in model_name.lower() for sub in ['image', 'vision', 'audio', 'video']):
                g4f_models.add(model_name)
    except ImportError:
        # *** FIX: Translated message ***
        print("Warning: Failed to import g4f.models. Model list may be incomplete.", file=sys.stderr)
        g4f_models = set()
    
    all_models = list(working_models.union(g4f_models))
    all_models = [m for m in all_models if m not in ['sldx-turbo', 'turbo']]
    return all_models


# =============================================================================
# === test_code() with CORRECTION in _expected_R ===
# =============================================================================

def test_code(code: str, config: Dict) -> Tuple[bool, str, Optional[Dict]]:
    """
    Testing code (LRX) with various vectors, including edge cases.
    """
    
    # --- Ground Truth implementations of LRX for verification ---
    def _expected_L(v):
        """Reference left shift"""
        if not v:
            return []
        return v[1:] + v[:1]

    def _expected_R(v):
        """Reference right shift"""
        if not v:
            return []
        # CORRECTION: v[-1:] (slice) instead of v['n-1'] (typo from repo)
        return v[-1:] + v[:-1] 

    def _expected_X(v):
        """Reference transposition"""
        if len(v) < 2:
            return v[:]
        # Create a new list, combining the swapped
        # first two elements and the rest of the list
        return [v[1], v[0]] + v[2:]
    # ---

    test_results = []
    all_success = True
    failing_cases = []
    total_time = 0.0
    exec_timeout = config['CONSTANTS']['EXEC_TIMEOUT']
    process = None  # For memory tracking
    temp_file_path = None


    def _run_single_test(vector, temp_file_path, exec_timeout):
        n = len(vector)
        arg = json.dumps(vector)
        start_time = perf_counter()
        child_process = None
        
        # Expected results (calculate before running)
        try:
            exp_l = _expected_L(vector)
            exp_r = _expected_R(vector)
            exp_x = _expected_X(vector)
        except Exception as e_gt:
            # This error is not in the LLM code, but in *our* ground truth code.
            # *** FIX: Translated message ***
            err_msg = f"Error in ground truth function: {e_gt}"
            print(err_msg, file=sys.stderr)
            traceback.print_exc()
            return {'n': n, 'success': False, 'error': err_msg, 'input': vector}, err_msg, False
        
        # Template dictionary for an error
        error_result_dict = {
            'n': n,
            'success': False,
            'time': 0.0,
            'input': vector,
            'L_result': None, 'expected_L': exp_l,
            'R_result': None, 'expected_R': exp_r,
            'X_result': None, 'expected_X': exp_x
        }

        try:
            child_process = subprocess.Popen(
                [sys.executable, temp_file_path, arg],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding='utf-8',
                errors='replace',
                preexec_fn=None  # Windows-compatible
            )
            stdout, stderr = child_process.communicate(timeout=exec_timeout)
            elapsed = perf_counter() - start_time
            error_result_dict['time'] = elapsed # Update time even in case of error

            if child_process.returncode != 0:
                err = stderr or 'Unknown error'
                error_result_dict['error'] = err
                return error_result_dict, err, False

            output = stdout.strip()
            if not output:
                err = 'No output'
                error_result_dict['error'] = err
                return error_result_dict, err, False

            try:
                parsed = json.loads(output)
                
                # --- NEW LRX VERIFICATION BLOCK ---
                l_res = parsed.get('L_result')
                r_res = parsed.get('R_result')
                x_res = parsed.get('X_result')

                success_l = (l_res == exp_l)
                success_r = (r_res == exp_r)
                success_x = (x_res == exp_x)
                success = success_l and success_r and success_x

                error_msg = None
                if not success:
                    errors = []
                    if not success_l: errors.append(f"L mismatch: got {l_res}, expected {exp_l}")
                    if not success_r: errors.append(f"R mismatch: got {r_res}, expected {exp_r}")
                    if not success_x: errors.append(f"X mismatch: got {x_res}, expected {exp_x}")
                    error_msg = "; ".join(errors)
                # --- END OF NEW BLOCK ---

                res_dict = {
                    'n': n,
                    'success': success,
                    'time': elapsed,
                    'input': vector,
                    'L_result': l_res,
                    'expected_L': exp_l,
                    'R_result': r_res,
                    'expected_R': exp_r,
                    'X_result': x_res,
                    'expected_X': exp_x
                }
                if error_msg:
                    res_dict['error'] = error_msg
                return res_dict, error_msg, success

            except json.JSONDecodeError as je:
                err = f'JSON parse error: {je}. Output was: {output}'
                error_result_dict['error'] = err
                return error_result_dict, err, False
            except KeyError as ke: 
                err = f'KeyError: {ke}. JSON output missing expected keys. Output was: {output}'
                error_result_dict['error'] = err
                # Fill with what we managed to parse
                error_result_dict['L_result'] = parsed.get('L_result')
                error_result_dict['R_result'] = parsed.get('R_result')
                error_result_dict['X_result'] = parsed.get('X_result')
                return error_result_dict, err, False

        except subprocess.TimeoutExpired:
            err = config['CONSTANTS']['ERROR_TIMEOUT']
            if child_process:
                child_process.kill()
            error_result_dict['error'] = err
            error_result_dict['time'] = exec_timeout # Time = timeout
            return error_result_dict, err, False
        finally:
            if child_process:
                child_process.wait()


    try:
        # Create temporary file for the code once
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as temp_file:
            temp_file.write(code)
            temp_file_path = temp_file.name

        # Start memory tracking process (parent)
        process = psutil.Process()
        start_mem = process.memory_info().rss / 1024  # KB
    except (ImportError, Exception):
        start_mem = None

    # --- UPDATED VECTORS for LRX ---
    specific_vectors = [
        [],                      # n=0
        [1],                     # n=1
        [1, 2],                  # n=2
        [1, 2, 3],               # n=3
        [3, 1, 2],
        [5, 2, 4, 1, 3],
        [48, 18, 44, 20, 16, 61, 26],
        [1, 2, 3, 4], # Base vector from prompt
    ]
    # ---

    # All vectors: specific + random for n=4 to 20
    all_vectors = specific_vectors + [
        [random.randint(0, n * 10) for _ in range(n)] for n in range(4, 21)
    ]

    for vector in all_vectors:
        res_dict, err_msg, single_success = _run_single_test(vector, temp_file_path, exec_timeout)
        total_time += res_dict['time']
        test_results.append(res_dict)

        if not single_success:
            all_success = False
            # --- UPDATED ERROR REPORT ---
            failing_case = {
                'n': res_dict['n'],
                'input': res_dict['input'],
                'error': res_dict.get('error', 'Unknown failure'),
                'L_result': res_dict.get('L_result'),
                'expected_L': res_dict.get('expected_L'),
                'R_result': res_dict.get('R_result'),
                'expected_R': res_dict.get('expected_R'),
                'X_result': res_dict.get('X_result'),
                'expected_X': res_dict.get('expected_X'),
            }
            # ---
            failing_cases.append(failing_case)

    # Clean up temp file
    if temp_file_path and os.path.exists(temp_file_path):
        os.unlink(temp_file_path)

    # End memory tracking
    try:
        if start_mem is not None and process:
            end_mem = process.memory_info().rss / 1024  # KB
            max_memory_kb = end_mem - start_mem
        else:
            max_memory_kb = None
    except Exception:
        max_memory_kb = None

    num_tests = len(test_results)
    summary = {
        'all_success': all_success,
        'total_time': total_time,
        'avg_time': total_time / num_tests if num_tests > 0 else 0,
        'max_memory_kb': max_memory_kb,
        'tests': test_results,
        'num_failing': len(failing_cases)
    }

    if all_success:
        return True, 'All tests passed', summary
    else:
        issue_str = json.dumps({'failing_cases': failing_cases}, ensure_ascii=False, indent=2)
        return False, issue_str, summary

# =============================================================================
# === END OF UPDATED test_code() FUNCTION ===
# =============================================================================


def llm_query(model: str, prompt: str, retries_config: Dict, config: Dict, progress_queue: queue.Queue, stage: str = None) -> Optional[str]:
    # Initialize local only for the patch (so it can write to the queue)
    local.current_model = model
    local.current_queue = progress_queue
    local.current_data = {'tried': [], 'errors': {}, 'success': None, 'model': model}
    local.current_stage = stage

    request_timeout = config['CONSTANTS']['REQUEST_TIMEOUT']

    # AnyProvider: simple call with retries
    for attempt in range(retries_config['max_retries'] + 1):
        try:
            response = g4f.ChatCompletion.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                provider=Provider.AnyProvider,
                timeout=request_timeout
            )
            if response and response.strip():
                return response.strip()
        except ModelNotFoundError as e:
            # IMPROVEMENT: Log if model not found
            # *** FIX: Translated message ***
            progress_queue.put((model, 'log', f'Error: ModelNotFoundError: {e}'))
            if len(e.args) > 1:
                local.current_data['tried'] = e.args[1]
            return None # Model not found, retry won't help
        except Exception as e:
            # IMPROVEMENT: Log any other g4f error (e.g., rate limit)
            # *** FIX: Translated message ***
            progress_queue.put((model, 'log', f'g4f Error (attempt {attempt+1}): {e}'))
            pass # Allow retry to work
        
        if attempt < retries_config['max_retries']:
            time.sleep(retries_config['backoff_factor'] * (2 ** attempt))

    return None

def process_model(model: str, config: Dict, progress_queue: queue.Queue) -> Dict:
    """
    Process one model: sequence of LLM queries, test, fix, refactor.
    CORRECTION: Removed `task` argument, as it caused a KeyError.
    """
    iterations = []
    current_code = None
    prev_code = None
    early_stop = False
    
    # Count total number of stages for progress bar
    # 1 (Initial) + 1 (Test/Fix) + 1 (Refactor1) + 1 (Test/Fix) + N*(Refactor + Test/Fix) + 1 (Final Test)
    num_loops = config['CONSTANTS']['NUM_REFACTOR_LOOPS']
    total_stages = 1 + 1 + 1 + 1 + (num_loops * 2) + 1
    current_stage_count = 0

    def update_progress(stage_name):
        nonlocal current_stage_count
        current_stage_count += 1
        # *** FIX: Translated message ***
        progress_queue.put((model, 'status', f'Stage: {stage_name} ({current_stage_count}/{total_stages})'))
        progress_queue.put((model, 'progress', (current_stage_count, total_stages)))

    def run_test(code_to_test, stage_name):
        """Internal function for testing and logging."""
        if not code_to_test or not code_to_test.strip():
            # *** FIX: Translated message ***
            progress_queue.put((model, 'log', f'Test {stage_name}: Skipped (no code).'))
            return False, "No code to test", None
            
        # *** FIX: Translated message ***
        progress_queue.put((model, 'log', f'Test {stage_name}: Running test_code...'))
        success, issue, summary = test_code(code_to_test, config)
        if success:
            progress_queue.put((model, 'log', f'Test {stage_name}: SUCCESS. {issue}'))
        else:
            progress_queue.put((model, 'log', f'Test {stage_name}: FAILED. Reason: {issue}'))
        return success, issue, summary

    def run_llm_query(prompt, stage_name, retries_key='FIX'):
        """Internal function for LLM query and logging."""
        # *** FIX: Translated message ***
        progress_queue.put((model, 'log', f'Stage: {stage_name}. Prompt:\n{prompt}'))
        retries_cfg = config['RETRIES'][retries_key]
        progress_queue.put((model, 'log', f'Calling llm_query with retries: {retries_cfg}'))
        
        response = llm_query(model, prompt, retries_cfg, config, progress_queue, stage_name)
        
        tried = local.current_data.get('tried', [])
        success_p = local.current_data.get('success', None)
        
        if response:
            cleaned = clean_code(response)
            # *** FIX: Translated message ***
            progress_queue.put((model, 'log', f'Received response (length: {len(response)}), cleaned (length: {len(cleaned)}):\n{cleaned}'))
            return cleaned, None, tried, success_p
        else:
            error_msg = config['CONSTANTS']['ERROR_NO_RESPONSE']
            # *** FIX: Translated message ***
            progress_queue.put((model, 'log', f'llm_query error: {error_msg}'))
            return None, error_msg, tried, success_p

    def add_iteration(stage, response, error, test_summary, tried, success_p):
        """Adds a record to the iteration history."""
        iterations.append({
            'providers_tried': tried,
            'success_provider': success_p,
            'stage': stage,
            'response': response, # Save the code (or None)
            'error': error, # LLM or test error
            'test_summary': test_summary # Result from test_code
        })

    # ---
    # START OF PROCESS
    # ---
    # *** FIX: Translated message ***
    progress_queue.put((model, 'log', f'=== STARTING MODEL PROCESSING: {model} ==='))
    
    # 1. Initial
    stage = config['STAGES']['INITIAL']
    update_progress(stage)
    
    # CORRECTION: Remove .format(task=task) to avoid KeyError: 'n-1'
    prompt = config['PROMPTS']['INITIAL'] 
    
    current_code, llm_error, tried, s_provider = run_llm_query(prompt, stage, 'INITIAL')
    add_iteration(stage, current_code, llm_error, None, tried, s_provider)
    
    if llm_error:
        # *** FIX: Translated message ***
        progress_queue.put((model, 'status', f'Error at stage: {stage}'))
        return {'model': model, 'iterations': iterations, 'final_code': None,
                'final_test': {'success': False, 'summary': None, 'issue': 'No initial response'}}

    # 2. Test & Fix (Initial)
    stage = config['STAGES']['FIX_INITIAL']
    update_progress(stage)
    success, issue, summary = run_test(current_code, stage)
    
    if not success:
        prompt = config['PROMPTS']['FIX'].format(code=current_code, error=issue)
        current_code, llm_error, tried, s_provider = run_llm_query(prompt, stage)
        add_iteration(stage, current_code, llm_error, summary, tried, s_provider) # Add FIX attempt
        if llm_error:
            early_stop = True # LLM error during fix = stop
    else:
        # Add record of successful test, even if FIX wasn't needed
        add_iteration(stage, current_code, None, summary, [], None) 

    if early_stop:
        # *** FIX: Translated message ***
        progress_queue.put((model, 'status', f'Error at stage: {stage}'))
        return {'model': model, 'iterations': iterations, 'final_code': current_code,
                'final_test': {'success': False, 'summary': summary, 'issue': f'LLM error during {stage}'}}

    # 3. Refactor (First)
    prev_code = current_code
    stage = config['STAGES']['REFACTOR_FIRST']
    update_progress(stage)
    prompt = config['PROMPTS']['REFACTOR_NO_PREV'].format(code=current_code)
    
    current_code, llm_error, tried, s_provider = run_llm_query(prompt, stage, 'INITIAL') # Use 'INITIAL' retries
    add_iteration(stage, current_code, llm_error, None, tried, s_provider)

    if llm_error:
        current_code = prev_code # Rollback if refactor failed
        # *** FIX: Translated message ***
        progress_queue.put((model, 'log', f'Error {stage}, rolling back to previous code version.'))
    
    # 4. Test & Fix (After Refactor 1)
    stage = config['STAGES']['FIX_AFTER_REFACTOR']
    update_progress(stage)
    success, issue, summary = run_test(current_code, stage)
    
    if not success:
        prompt = config['PROMPTS']['FIX'].format(code=current_code, error=issue)
        current_code, llm_error, tried, s_provider = run_llm_query(prompt, stage)
        add_iteration(stage, current_code, llm_error, summary, tried, s_provider)
        if llm_error:
            early_stop = True
    else:
        add_iteration(stage, current_code, None, summary, [], None)

    if early_stop:
        # *** FIX: Translated message ***
        progress_queue.put((model, 'status', f'Error at stage: {stage}'))
        return {'model': model, 'iterations': iterations, 'final_code': current_code,
                'final_test': {'success': False, 'summary': summary, 'issue': f'LLM error during {stage}'}}

    # 5. Refactor Loops (N times)
    for i in range(config['CONSTANTS']['NUM_REFACTOR_LOOPS']):
        if not current_code or not current_code.strip():
            # *** FIX: Translated message ***
            progress_queue.put((model, 'log', f'Skipping refactor loop {i+1} (no code).'))
            update_progress(f'loop {i+1} refactor (skip)') # Skip 2 stages
            update_progress(f'loop {i+1} fix (skip)')
            continue
            
        # 5a. Refactor
        stage = f"{config['STAGES']['REFACTOR']}_{i+1}"
        update_progress(stage)
        
        prompt = config['PROMPTS']['REFACTOR'].format(code=current_code, prev=prev_code)
        prev_code = current_code # Save for next loop
        
        current_code, llm_error, tried, s_provider = run_llm_query(prompt, stage, 'INITIAL')
        add_iteration(stage, current_code, llm_error, None, tried, s_provider)

        if llm_error:
            current_code = prev_code # Rollback
            # *** FIX: Translated message ***
            progress_queue.put((model, 'log', f'Error {stage}, rolling back to previous code version.'))
        
        # 5b. Test & Fix
        stage = f"{config['STAGES']['FIX_LOOP']}_{i+1}"
        update_progress(stage)
        success, issue, summary = run_test(current_code, stage)
        
        if not success:
            prompt = config['PROMPTS']['FIX'].format(code=current_code, error=issue)
            current_code, llm_error, tried, s_provider = run_llm_query(prompt, stage)
            add_iteration(stage, current_code, llm_error, summary, tried, s_provider)
            if llm_error:
                # *** FIX: Translated message ***
                progress_queue.put((model, 'status', f'Error at stage: {stage}, STOPPING LOOP'))
                break # Break refactor loop if LLM couldn't fix
        else:
            add_iteration(stage, current_code, None, summary, [], None)

    # 6. Final Test
    stage = 'final_test'
    update_progress(stage)
    success, issue, summary = run_test(current_code, stage)
    
    # Add final record (duplicates final_test in response, but useful for history)
    add_iteration(stage, current_code, None if success else issue, summary, [], None)

    if success:
        # *** FIX: Translated message ***
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
            # *** FIX: Translated message ***
            print(f"Error creating folder {folder}: {e}", file=sys.stderr)
            return
    path = os.path.join(folder, filename)
    try:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
    except Exception as e:
        # *** FIX: Translated message ***
        print(f"Error saving file {path}: {e}", file=sys.stderr)

def main():
    """Main function: load models, start threads, process results."""
    # *** FIX: Translated message ***
    print("Loading model list...")
    try:
        models = get_models_list(CONFIG)
        if not models:
            print("No models found. Check URLS and g4f.models.", file=sys.stderr)
            return
        print(f"Found {len(models)} unique models for testing.")
        # print(models) # Uncomment to see the list
    except Exception as e:
        print(f"Failed to load model list: {e}", file=sys.stderr)
        traceback.print_exc()
        return

    # Ensure the results folder exists
    intermediate_folder = CONFIG['CONSTANTS']['INTERMEDIATE_FOLDER']
    if not os.path.exists(intermediate_folder):
        try:
            os.makedirs(intermediate_folder)
        except OSError as e:
            # *** FIX: Translated message ***
            print(f"Failed to create folder {intermediate_folder}: {e}", file=sys.stderr)
            return

    progress_queue = queue.Queue()
    all_results = {}
    
    # Limit number of models for example (set to -1 for all)
    MAX_MODELS_TO_TEST = -1 # -1 for all, 10 for quick test
    
    if MAX_MODELS_TO_TEST > 0:
        models_to_test = models[:MAX_MODELS_TO_TEST]
        # *** FIX: Translated message ***
        print(f"--- STARTING TEST (Limited to {len(models_to_test)} models) ---")
    else:
        models_to_test = models
        # *** FIX: Translated message ***
        print(f"--- STARTING TEST (All {len(models_to_test)} models) ---")


    try:
        with ThreadPoolExecutor(max_workers=CONFIG['CONSTANTS']['MAX_WORKERS']) as executor:
            # CORRECTION: Remove `task_description` from call
            futures = {executor.submit(process_model, model, CONFIG, progress_queue): model for model in models_to_test}
            
            completed_count = 0
            total_count = len(futures)
            
            start_time_main = perf_counter()

            while completed_count < total_count:
                # 1. Check completed tasks
                done_futures = [f for f in futures if f.done()]
                
                for future in done_futures:
                    model = futures.pop(future)
                    completed_count += 1
                    try:
                        result = future.result()
                        all_results[model] = result
                        
                        final_success = result.get('final_test', {}).get('success', False)
                        # *** FIX: Translated message ***
                        status_str = "SUCCESS" if final_success else "FAILED"

                        # Save code if it exists
                        if result.get('final_code'):
                            code_filename = f"{model.replace('/', '_')}_final.py"
                            code_path = os.path.join(intermediate_folder, code_filename)
                            try:
                                with open(code_path, 'w', encoding='utf-8') as f:
                                    f.write(result['final_code'])
                            except Exception as e:
                                # *** FIX: Translated message ***
                                print(f"Error saving code for {model}: {e}", file=sys.stderr)
                                
                        # *** FIX: Translated message ***
                        print(f"--- ({completed_count}/{total_count}) COMPLETED: {model} [Status: {status_str}] ---")
                        
                    except Exception as e:
                        # IMPROVEMENT: Print full traceback for CRITICAL ERROR
                        # *** FIX: Translated message ***
                        print(f"--- ({completed_count+1}/{total_count}) CRITICAL ERROR (Executor): {model} ---")
                        tb_str = traceback.format_exc()
                        print(tb_str, file=sys.stderr) # Print full traceback
                        all_results[model] = {'error': str(e), 'traceback': tb_str, 'iterations': [], 'final_code': None, 'final_test': {'success': False, 'summary': None, 'issue': str(e)}}
                    
                    # Intermediate save
                    if completed_count % CONFIG['CONSTANTS']['N_SAVE'] == 0 or completed_count == total_count:
                        save_results(all_results, intermediate_folder, f"intermediate_results_{completed_count}.json")

                # 2. Process log queue (with timeout to avoid blocking)
                try:
                    while not progress_queue.empty():
                        model, type, message = progress_queue.get_nowait()
                        # Can be uncommented for VERY detailed logs
                        # if type == 'log':
                        #     print(f"LOG [{model}]: {message}")
                        # elif type == 'status':
                        #     print(f"STATUS [{model}]: {message}")
                
                except queue.Empty:
                    pass # Queue is empty, this is normal
                
                time.sleep(0.2) # Short pause to avoid high CPU usage

    except KeyboardInterrupt:
        # *** FIX: Translated message ***
        print("\nStopping at user request... (Waiting for current threads to finish)")
    finally:
        end_time_main = perf_counter()
        total_time_main = end_time_main - start_time_main
        # *** FIX: Translated message ***
        print("--- TESTING FINISHED ---")
        print(f"Total execution time: {total_time_main:.2f} sec.")
        
        # Final save
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        final_filename = f"final_results_{timestamp}.json"
        save_results(all_results, intermediate_folder, final_filename)
        print(f"Final results saved to: {os.path.join(intermediate_folder, final_filename)}")
        
        # Summary
        success_count = sum(1 for res in all_results.values() if res.get('final_test', {}).get('success', False))
        fail_count = len(all_results) - success_count
        print(f"Totals: {success_count} Successful, {fail_count} Failed.")


if __name__ == "__main__":
    main()
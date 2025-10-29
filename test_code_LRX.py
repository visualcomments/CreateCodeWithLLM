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
    # Find first block ```python\n... (up to next ``` or end)
    
    # === ИСПРАВЛЕНИЕ 1 (Было: match = re.search(r'...) ===
    match = re.search(r'```(?:python\n)?(.*?)\n```', code, re.DOTALL | re.MULTILINE)

    if match:
        code = match.group(1)
    # Alternative: if block without closing ```, search from first ``` to end
    else:
        # === ИСПРАВЛЕНИЕ 2 (Было: match = re.search(r'...) ===
        match = re.search(r'```(?:python\n)?(.*?)$', code, re.DOTALL | re.MULTILINE)

        if match:
            code = match.group(1)

    # Step 3: Final regex cleanup (for nested markdown)
    # Remove ```python block at start
    # === ИСПРАВЛЕНИЕ 3 (Было: code = re.sub(r'^...) ===
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


# =============================================================================
# === CONFIG: PROMPTS UPDATED FOR SORTING TASK ===
# =============================================================================

CONFIG = {
    # Section with URLs for downloading data about working models
    'URLS': {
        # URL of the file with test results for working g4f models
        'WORKING_RESULTS': '[https://raw.githubusercontent.com/maruf009sultan/g4f-working/refs/heads/main/working/working_results.txt](https://raw.githubusercontent.com/maruf009sultan/g4f-working/refs/heads/main/working/working_results.txt)'
    },

    # Section with prompts for various stages of interaction with LLM
    'PROMPTS': {
        # This is the NEW task (Prompt 2)
        'INITIAL': r"""
Task: Implement a sorting algorithm that sorts a given vector using ONLY allowed moves (L, R, X).

    Input: A vector a of length n (0-indexed) containing distinct integers (permutations are assumed for simplicity).

    Allowed moves:
    L: Left cyclic shift — shifts all elements one position to the left, with the first element moving to the end. Example: [1,2,3,4] -> [2,3,4,1].
    R: Right cyclic shift — shifts all elements one position to the right, with the last element moving to the beginning. Example: [1,2,3,4] -> [4,1,2,3].
    X: Transposition of the first two elements — swaps the elements at positions 0 and 1. Example: [1,2,3,4] -> [2,1,3,4].

    Strict constraints:
    No other operations, slicing, built-in sorting functions, or creating new arrays are allowed (except for a copy to simulate sorting).
    All moves must be appended to the moves list immediately after performing them (as strings: 'L', 'R', or 'X').
    Applying the sequence of moves sequentially to a copy of the input vector must yield a fully sorted ascending array.
    Moves can be used multiple times as needed during the sorting process.
    The sorting algorithm must continue applying moves until the array is fully sorted.
    The algorithm should be efficient and find a sequence of moves that sorts the array (it represents a path in the Cayley graph generated by these moves).

    Description of moves (Cayley graph):
    These moves generate a Cayley graph where vertices are all possible permutations of the vector (for n elements, there are n! vertices).
    Edges correspond to applying one of the generators: L (left shift), R (right shift), or X (swap first two).
    The graph is undirected if moves are invertible (note: L and R are inverses of each other; X is its own inverse).
    The sorting task is equivalent to finding a path from the input permutation to the sorted (identity) permutation in this Cayley graph.
    Your algorithm must compute such a path as a sequence of moves.

    Requirements:
    Implement a function solve(vector) that returns a tuple (moves, sorted_array):
        moves is a list of strings representing all moves performed (e.g., ['L', 'X', 'R', ...]).
        sorted_array is the final sorted array after applying all moves to a copy of the input vector.
    The algorithm must be a sorting algorithm that uses only L, R, X to transform the vector into sorted order (e.g., an adaptation of a search-based sorter like BFS for shortest path in the Cayley graph, or a heuristic sorter).
    Include a CLI interface:
        When the script is executed directly, it should accept a vector as a command-line argument (parse sys.argv[1] as JSON). Use [3, 1, 2] as a fallback if no arg is given.
        The output should be a JSON object with keys "moves" and "sorted_array".
    Include a minimal example in the main block for quick testing.
    The code must be fully self-contained and executable without external dependencies.
    JSON output must always be structured and parseable for automated testing.

    Example expected usage:

        python solve_module.py "[3,1,2,5,4]"

    Example output (hypothetical for input [3,1,2]):
    {
        "moves": ["X", "L", "R", "X"],
        "sorted_array": [1,2,3]
    }
""",
        # NEW FIX prompt for the sorting task
        'FIX': r"""
You are a Python debugging assistant. The following code, intended to be an LRX sorting algorithm, did not work correctly.
Fix it to meet all requirements.

Code:
{code}

Issue:
{error}

⚠️ Requirements:
• Must implement a function `solve(vector)`.
• `solve(vector)` must return a tuple (moves, sorted_array).
• `moves` is a list of strings (e.g., ['L', 'X']).
• `sorted_array` is the final, correctly sorted list.
• CLI: `import json`, `import sys`; parse `sys.argv[1]` as JSON (use `try-except`, fallback to [3, 1, 2]).
• Call `moves, sorted_array = solve(vector)`.
• Print ONLY `json.dumps({{"moves": moves, "sorted_array": sorted_array}})`.
• The sorting logic must use *only* L, R, X operations internally to modify the list state.
• Self-contained, executable, immediately correct.
• Only code in response, no extra prints or Markdown.
""",
        # NEW REFACTOR prompt for the sorting task
        'REFACTOR_NO_PREV': r"""
You are an expert Python programmer. Refactor the following LRX sorting code:

{code}

⚠️ Goals:
• Improve the sorting algorithm's efficiency (e.g., using BFS for shortest path, or a more direct heuristic).
• Ensure `solve(vector)` returns (moves, sorted_array).
• Preserve CLI: parse `sys.argv[1]` (fallback [3, 1, 2]), print only `json.dumps({{"moves": moves, "sorted_array": sorted_array}})`.
• Minimal example in __main__ must print JSON only.
• Fully executable, immediately correct.
• Only code in response, no explanations or Markdown.
""",
        # NEW REFACTOR prompt for the sorting task
        'REFACTOR': r"""
You are an expert Python programmer. Compare the current and previous versions of this LRX sorter and perform a full refactor:

Current code:
{code}

Previous version:
{prev}

⚠️ Goals:
• Improve the sorting algorithm's efficiency (e.g., using BFS for shortest path, or a more direct heuristic).
• Ensure `solve(vector)` returns (moves, sorted_array).
• Preserve CLI: parse `sys.argv[1]` (fallback [3, 1, 2]), print only `json.dumps({{"moves": moves, "sorted_array": sorted_array}})`.
• Code must pass verification (e.g., solve([3,1,2]) results in [1,2,3] and the moves list must be valid).
• Only code in response, no explanations or Markdown.
"""
    },

    # Section with retry settings for different request types
    'RETRIES': {
        'INITIAL': {'max_retries': 1, 'backoff_factor': 1.0},
        'FIX': {'max_retries': 3, 'backoff_factor': 2.0}
    },

    # Section with system constants
    'CONSTANTS': {
        'DELIMITER_MODEL': '|',
        'MODEL_TYPE_TEXT': 'text',
        'REQUEST_TIMEOUT': 10,
        'N_SAVE': 100,
        'MAX_WORKERS': 10,
        'EXEC_TIMEOUT': 10, # Increased timeout for sorting
        'ERROR_TIMEOUT': 'Timeout expired — the sorting algorithm is too slow or stuck in an infinite loop.',
        'ERROR_NO_RESPONSE': 'No response from model',
        'NUM_REFACTOR_LOOPS': 3,
        'INTERMEDIATE_FOLDER': 'results'
    },

    # Section with stage names for logs and statuses
    'STAGES': {
        'INITIAL': 'initial_response',
        'FIX_INITIAL': 'fix_before_refactor',
        'REFACTOR_FIRST': 'refactor_first_response',
        'FIX_AFTER_REFACTOR': 'fix_after_revector',
        'REFACTOR': 'refactor_loop',
        'FIX_LOOP': 'fix_loop'
    }
}
# =============================================================================
# === END OF CONFIG UPDATE ===
# =============================================================================


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
        print("Warning: Failed to import g4f.models. Model list may be incomplete.", file=sys.stderr)
        g4f_models = set()
    
    all_models = list(working_models.union(g4f_models))
    all_models = [m for m in all_models if m not in ['sldx-turbo', 'turbo']]
    return all_models


# =============================================================================
# === test_code() REWRITTEN FOR SORTING TASK ===
# =============================================================================

def test_code(code: str, config: Dict) -> Tuple[bool, str, Optional[Dict]]:
    """
    Testing code (LRX Sorter) with various vectors, including edge cases.
    This function is now completely different. It verifies:
    1. The subprocess returns valid JSON: {"moves": [...], "sorted_array": [...]}
    2. The "sorted_array" is actually sorted.
    3. Applying the "moves" to the original vector produces the "sorted_array".
    """
    
    # --- Ground Truth: Move application functions for verification ---
    def _apply_move(v_list: List, move: str) -> List:
        """Applies a single move to a list and returns a new list."""
        n = len(v_list)
        if n == 0:
            return []
        
        if move == 'L':
            return v_list[1:] + v_list[:1]
        
        if move == 'R':
            return v_list[-1:] + v_list[:-1]
            
        if move == 'X':
            if n < 2:
                return v_list[:]
            return [v_list[1], v_list[0]] + v_list[2:]
            
        # Invalid move
        raise ValueError(f"Unknown move '{move}'")
    # ---

    test_results = []
    all_success = True
    failing_cases = []
    total_time = 0.0
    exec_timeout = config['CONSTANTS']['EXEC_TIMEOUT']
    process = None
    temp_file_path = None


    def _run_single_test(vector, temp_file_path, exec_timeout):
        n = len(vector)
        arg = json.dumps(vector)
        start_time = perf_counter()
        child_process = None
        
        # Expected results (calculate before running)
        try:
            expected_sorted_list = sorted(vector)
        except Exception as e_gt:
            err_msg = f"Error in ground truth function (sorted()): {e_gt}"
            print(err_msg, file=sys.stderr)
            traceback.print_exc()
            return {'n': n, 'success': False, 'error': err_msg, 'input': vector}, err_msg, False
        
        # Template dictionary for an error
        error_result_dict = {
            'n': n,
            'success': False,
            'time': 0.0,
            'input': vector,
            'moves_result': None,
            'array_result': None,
            'expected_array': expected_sorted_list,
        }

        try:
            child_process = subprocess.Popen(
                [sys.executable, temp_file_path, arg],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding='utf-8',
                errors='replace',
                preexec_fn=None
            )
            stdout, stderr = child_process.communicate(timeout=exec_timeout)
            elapsed = perf_counter() - start_time
            error_result_dict['time'] = elapsed

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
                
                # --- NEW SORTER VERIFICATION BLOCK ---
                moves_result = parsed.get('moves')
                array_result = parsed.get('sorted_array')
                
                error_result_dict['moves_result'] = moves_result
                error_result_dict['array_result'] = array_result

                # 1. Check if array is sorted
                if array_result != expected_sorted_list:
                    err = f"Array not sorted. Got {array_result}, expected {expected_sorted_list}"
                    error_result_dict['error'] = err
                    return error_result_dict, err, False

                # 2. Check if moves is a list (or at least iterable)
                if not isinstance(moves_result, list):
                    err = f"Moves is not a list. Got {type(moves_result)}"
                    error_result_dict['error'] = err
                    return error_result_dict, err, False

                # 3. Verify moves
                try:
                    temp_v = vector[:]
                    for move in moves_result:
                        temp_v = _apply_move(temp_v, move)
                    
                    if temp_v != array_result:
                        err = f"Moves do not produce sorted array. Applying moves {moves_result} to {vector} produced {temp_v}, but expected {array_result}"
                        error_result_dict['error'] = err
                        return error_result_dict, err, False
                        
                except Exception as e_apply:
                    err = f"Error while applying moves: {e_apply}. Moves were: {moves_result}"
                    error_result_dict['error'] = err
                    return error_result_dict, err, False
                
                # All checks passed
                success = True
                error_msg = None
                # --- END OF NEW BLOCK ---

                res_dict = {
                    'n': n,
                    'success': success,
                    'time': elapsed,
                    'input': vector,
                    'moves_result': moves_result,
                    'array_result': array_result,
                    'expected_array': expected_sorted_list,
                    'num_moves': len(moves_result)
                }
                return res_dict, error_msg, success

            except json.JSONDecodeError as je:
                err = f'JSON parse error: {je}. Output was: {output}'
                error_result_dict['error'] = err
                return error_result_dict, err, False
            except KeyError as ke: 
                err = f'KeyError: {ke}. JSON output missing "moves" or "sorted_array". Output was: {output}'
                error_result_dict['error'] = err
                return error_result_dict, err, False

        except subprocess.TimeoutExpired:
            err = config['CONSTANTS']['ERROR_TIMEOUT']
            if child_process:
                child_process.kill()
            error_result_dict['error'] = err
            error_result_dict['time'] = exec_timeout
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

    # --- UPDATED VECTORS for Sorting Task ---
    # We use smaller N because sorting is N! complexity
    specific_vectors = [
        [],                      # n=0
        [1],                     # n=1
        [1, 2],                  # n=2 (already sorted)
        [2, 1],                  # n=2 (needs X)
        [1, 2, 3],               # n=3 (already sorted)
        [3, 1, 2],               # n=3
        [2, 3, 1],               # n=3
        [3, 2, 1],               # n=3 (reverse)
        [4, 1, 3, 2],            # n=4
        [1, 3, 2, 4],            # n=4
        [5, 4, 3, 2, 1],         # n=5 (reverse)
        [1, 5, 2, 4, 3],         # n=5
        [6, 1, 2, 3, 4, 5]       # n=6 (testing timeout)
    ]
    # ---
    
    # We remove the random loop for n=4-20 as it's not feasible
    # for a N! sorting problem test.
    all_vectors = specific_vectors

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
                'moves_result': res_dict.get('moves_result'),
                'array_result': res_dict.get('array_result'),
                'expected_array': res_dict.get('expected_array'),
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
        # Only show the first failing case for brevity
        issue_str = json.dumps({'failing_cases': failing_cases[:1]}, ensure_ascii=False, indent=2)
        return False, issue_str, summary

# =============================================================================
# === END OF REWRITTEN test_code() FUNCTION ===
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
            progress_queue.put((model, 'log', f'Error: ModelNotFoundError: {e}'))
            if len(e.args) > 1:
                local.current_data['tried'] = e.args[1]
            return None # Model not found, retry won't help
        except Exception as e:
            progress_queue.put((model, 'log', f'g4f Error (attempt {attempt+1}): {e}'))
            pass # Allow retry to work
        
        if attempt < retries_config['max_retries']:
            time.sleep(retries_config['backoff_factor'] * (2 ** attempt))

    return None

def process_model(model: str, config: Dict, progress_queue: queue.Queue) -> Dict:
    """
    Process one model: sequence of LLM queries, test, fix, refactor.
    """
    iterations = []
    current_code = None
    prev_code = None
    early_stop = False
    
    # Count total number of stages for progress bar
    num_loops = config['CONSTANTS']['NUM_REFACTOR_LOOPS']
    total_stages = 1 + 1 + 1 + 1 + (num_loops * 2) + 1
    current_stage_count = 0

    def update_progress(stage_name):
        nonlocal current_stage_count
        current_stage_count += 1
        progress_queue.put((model, 'status', f'Stage: {stage_name} ({current_stage_count}/{total_stages})'))
        progress_queue.put((model, 'progress', (current_stage_count, total_stages)))

    def run_test(code_to_test, stage_name):
        """Internal function for testing and logging."""
        if not code_to_test or not code_to_test.strip():
            progress_queue.put((model, 'log', f'Test {stage_name}: Skipped (no code).'))
            return False, "No code to test", None
            
        progress_queue.put((model, 'log', f'Test {stage_name}: Running test_code...'))
        # NOTE: This now calls the NEW test_code() function
        success, issue, summary = test_code(code_to_test, config)
        if success:
            progress_queue.put((model, 'log', f'Test {stage_name}: SUCCESS. {issue}'))
        else:
            progress_queue.put((model, 'log', f'Test {stage_name}: FAILED. Reason: {issue}'))
        return success, issue, summary

    def run_llm_query(prompt, stage_name, retries_key='FIX'):
        """Internal function for LLM query and logging."""
        progress_queue.put((model, 'log', f'Stage: {stage_name}. Prompt:\n{prompt[:500]}...'))
        retries_cfg = config['RETRIES'][retries_key]
        progress_queue.put((model, 'log', f'Calling llm_query with retries: {retries_cfg}'))
        
        response = llm_query(model, prompt, retries_cfg, config, progress_queue, stage_name)
        
        tried = local.current_data.get('tried', [])
        success_p = local.current_data.get('success', None)
        
        if response:
            cleaned = clean_code(response)
            progress_queue.put((model, 'log', f'Received response (length: {len(response)}), cleaned (length: {len(cleaned)}):\n{cleaned[:500]}...'))
            return cleaned, None, tried, success_p
        else:
            error_msg = config['CONSTANTS']['ERROR_NO_RESPONSE']
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
    progress_queue.put((model, 'log', f'=== STARTING MODEL PROCESSING: {model} ==='))
    
    # 1. Initial
    stage = config['STAGES']['INITIAL']
    update_progress(stage)
    
    prompt = config['PROMPTS']['INITIAL'] 
    
    current_code, llm_error, tried, s_provider = run_llm_query(prompt, stage, 'INITIAL')
    add_iteration(stage, current_code, llm_error, None, tried, s_provider)
    
    if llm_error:
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
        progress_queue.put((model, 'status', f'Error at stage: {stage}'))
        return {'model': model, 'iterations': iterations, 'final_code': current_code,
                'final_test': {'success': False, 'summary': summary, 'issue': f'LLM error during {stage}'}}

    # 5. Refactor Loops (N times)
    for i in range(config['CONSTANTS']['NUM_REFACTOR_LOOPS']):
        if not current_code or not current_code.strip():
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
    """Main function: load models, start threads, process results."""
    print("Loading model list...")
    try:
        models = get_models_list(CONFIG)
        if not models:
            print("No models found. Check URLS and g4f.models.", file=sys.stderr)
            return
        print(f"Found {len(models)} unique models for testing.")
    except Exception as e:
        print(f"Failed to load model list: {e}", file=sys.stderr)
        traceback.print_exc()
        return

    intermediate_folder = CONFIG['CONSTANTS']['INTERMEDIATE_FOLDER']
    if not os.path.exists(intermediate_folder):
        try:
            os.makedirs(intermediate_folder)
        except OSError as e:
            print(f"Failed to create folder {intermediate_folder}: {e}", file=sys.stderr)
            return

    progress_queue = queue.Queue()
    all_results = {}
    
    MAX_MODELS_TO_TEST = -1 # -1 for all, 10 for quick test
    
    if MAX_MODELS_TO_TEST > 0:
        models_to_test = models[:MAX_MODELS_TO_TEST]
        print(f"--- STARTING TEST (Limited to {len(models_to_test)} models) ---")
    else:
        models_to_test = models
        print(f"--- STARTING TEST (All {len(models_to_test)} models) ---")


    try:
        with ThreadPoolExecutor(max_workers=CONFIG['CONSTANTS']['MAX_WORKERS']) as executor:
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
                        status_str = "SUCCESS" if final_success else "FAILED"

                        # Save code if it exists
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
        print("\nStopping at user request... (Waiting for current threads to finish)")
    finally:
        end_time_main = perf_counter()
        total_time_main = end_time_main - start_time_main
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
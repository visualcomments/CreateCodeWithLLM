﻿import requests
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
import traceback

# Enhanced import handling with better error management
try:
    import g4f.providers.retry_provider as retry_mod
    OriginalRotatedProvider = retry_mod.RotatedProvider
except ImportError:
    print("Не удалось импортировать g4f.providers.retry_provider. Используем заглушку.", file=sys.stderr)
    class OriginalRotatedProvider:
        pass

try:
    import g4f
    from g4f import Provider
    from g4f.errors import ModelNotFoundError
except ImportError as e:
    print(f"Critical: g4f import failed: {e}", file=sys.stderr)
    sys.exit(1)

import threading
import queue

local = threading.local()

def clean_code(code: str) -> str:
    """Enhanced code cleaning with better markdown and JSON handling"""
    if not code or not isinstance(code, str):
        return ""
        
    original_len = len(code)
    
    # Step 1: Try JSON parsing (OpenAI-style responses)
    content_from_json = None
    try:
        data = json.loads(code)
        if isinstance(data, dict):
            # Handle various JSON response formats
            if 'choices' in data and len(data['choices']) > 0:
                content = data['choices'][0].get('message', {}).get('content', '')
                if content:
                    content_from_json = content
            elif 'content' in data:
                content_from_json = data['content']
            elif 'response' in data:
                content_from_json = data['response']
                
            if content_from_json:
                code = content_from_json
    except (json.JSONDecodeError, KeyError, IndexError, TypeError):
        pass

    # Step 2: Extract code from markdown code blocks
    code_blocks = re.findall(r'```(?:python)?\s*(.*?)\s*```', code, re.DOTALL)
    if code_blocks:
        code = code_blocks[0]
    else:
        # Try to find code between ```python and ``` or from first ``` to end
        match = re.search(r'```(?:python)?\s*\n(.*?)(?=\n\s*```|$)', code, re.DOTALL)
        if match:
            code = match.group(1)

    # Step 3: Final cleaning
    code = re.sub(r'^```(?:python)?\s*\n?', '', code, flags=re.MULTILINE)
    code = re.sub(r'\n\s*```\s*$', '', code, flags=re.MULTILINE)
    code = re.sub(r'^\s*\n+', '', code)
    code = re.sub(r'\s*\n+$', '\n', code)
    
    return code.strip()

# Custom Rotated Provider with better tracking
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
            local.current_queue.put((local.current_model, 'log', f'1) Found providers: {[p.__name__ if hasattr(p, "__name__") else str(p) for p in self.providers]}'))

        if not self.providers:
            raise ModelNotFoundError(f"No providers found for model {model}", [])

        for provider_class in self.providers:
            p = None
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
                continue
                
        raise ModelNotFoundError(f"No working provider for model {model}", current_data['tried'])

# Apply monkey patch
try:
    retry_mod.RotatedProvider = TrackedRotated
except NameError:
    print("Не удалось применить Monkey-patch для RotatedProvider", file=sys.stderr)

# Enhanced configuration
CONFIG = {
    'URLS': {
        'WORKING_RESULTS': 'https://raw.githubusercontent.com/maruf009sultan/g4f-working/main/working/working_results.txt'
    },

    'PROMPTS': {
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
• Print only one JSON object: `{"L_result": L(vector), "R_result": R(vector), "X_result": X(vector)}`.
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
• CLI: `import json`; parse `sys.argv[1]` with fallback [1, 2, 3, 4]; print only `json.dumps({"L_result": L(vector), "R_result": R(vector), "X_result": X(vector)})`.
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
• Preserve CLI: parse `sys.argv[1]` as JSON with fallback [1, 2, 3, 4], print only `json.dumps({"L_result": L(vector), "R_result": R(vector), "X_result": X(vector)})`.
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
• Preserve CLI: parse `sys.argv[1]` as JSON with fallback [1, 2, 3, 4]; print only `json.dumps({"L_result": L(vector), "R_result": R(vector), "X_result": X(vector)})`.
• Code must pass verification (e.g., L([1,2,3]) == [2,3,1], R([1,2,3]) == [3,1,2], X([1,2,3]) == [2,1,3]).
• Only code in response, no explanations or Markdown.
"""
    },

    'RETRIES': {
        'INITIAL': {'max_retries': 2, 'backoff_factor': 1.5},
        'FIX': {'max_retries': 3, 'backoff_factor': 2.0}
    },

    'CONSTANTS': {
        'DELIMITER_MODEL': '|',
        'MODEL_TYPE_TEXT': 'text',
        'REQUEST_TIMEOUT': 15,
        'N_SAVE': 50,
        'MAX_WORKERS': 8,
        'EXEC_TIMEOUT': 10,
        'ERROR_TIMEOUT': 'Timeout expired — the program likely entered an infinite loop.',
        'ERROR_NO_RESPONSE': 'No response from model',
        'NUM_REFACTOR_LOOPS': 2,
        'INTERMEDIATE_FOLDER': 'промежуточные результаты'
    },

    'STAGES': {
        'INITIAL': 'первичный_ответ',
        'FIX_INITIAL': 'исправление_до_рефакторинга',
        'REFACTOR_FIRST': 'ответ_от_рефакторинга',
        'FIX_AFTER_REFACTOR': 'исправление_после_рефакторинга',
        'REFACTOR': 'рефакторинг_в_цикле',
        'FIX_LOOP': 'исправление_в_цикле'
    }
}

def get_models_list(config: Dict) -> List[str]:
    """Enhanced model list retrieval with better error handling and fallbacks"""
    working_models = set()
    url_txt = config['URLS']['WORKING_RESULTS']
    
    # Fix URL formatting issue
    if url_txt.startswith('[') and url_txt.endswith(']'):
        url_txt = url_txt[1:-1]  # Remove brackets if present
    
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
    
    # Get models from g4f with better filtering
    try:
        from g4f.models import Model
        all_g4f_models = Model.__all__()
        g4f_models = set()
        excluded_keywords = ['flux', 'image', 'vision', 'audio', 'video', 'dall', 'sdxl', 'stable-diffusion']
        
        for model_name in all_g4f_models:
            model_lower = model_name.lower()
            if not any(keyword in model_lower for keyword in excluded_keywords):
                g4f_models.add(model_name)
    except Exception as e:
        print(f"Warning: Failed to import g4f models: {e}", file=sys.stderr)
        g4f_models = set()
    
    all_models = list(working_models.union(g4f_models))
    
    # Filter out problematic models
    excluded_models = ['sldx-turbo', 'turbo', 'gpt-4.5', 'gpt-4.1']  # Often cause issues
    all_models = [m for m in all_models if m not in excluded_models]
    
    print(f"Found {len(all_models)} total models ({len(working_models)} from URL, {len(g4f_models)} from g4f)")
    return all_models

def test_code(code: str, config: Dict) -> Tuple[bool, str, Optional[Dict]]:
    """Enhanced testing with better error handling and performance"""
    
    def _expected_L(v):
        return v[1:] + v[:1] if v else []
    
    def _expected_R(v):
        return v[-1:] + v[:-1] if v else []
    
    def _expected_X(v):
        if len(v) < 2:
            return v[:]
        return [v[1], v[0]] + v[2:]

    test_results = []
    all_success = True
    failing_cases = []
    total_time = 0.0
    exec_timeout = config['CONSTANTS']['EXEC_TIMEOUT']
    temp_file_path = None

    def _run_single_test(vector, temp_file_path, exec_timeout):
        arg = json.dumps(vector)
        start_time = perf_counter()
        child_process = None
        
        try:
            exp_l = _expected_L(vector)
            exp_r = _expected_R(vector)
            exp_x = _expected_X(vector)
        except Exception as e_gt:
            err_msg = f"Reference function error: {e_gt}"
            return {'n': len(vector), 'success': False, 'error': err_msg, 'input': vector}, err_msg, False

        error_result_dict = {
            'n': len(vector), 'success': False, 'time': 0.0, 'input': vector,
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
                errors='replace'
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

                res_dict = {
                    'n': len(vector), 'success': success, 'time': elapsed, 'input': vector,
                    'L_result': l_res, 'expected_L': exp_l,
                    'R_result': r_res, 'expected_R': exp_r,
                    'X_result': x_res, 'expected_X': exp_x
                }
                if error_msg:
                    res_dict['error'] = error_msg
                return res_dict, error_msg, success

            except (json.JSONDecodeError, KeyError) as e:
                err = f'Output parsing error: {e}. Output: {output[:200]}'
                error_result_dict['error'] = err
                return error_result_dict, err, False

        except subprocess.TimeoutExpired:
            err = config['CONSTANTS']['ERROR_TIMEOUT']
            if child_process:
                child_process.kill()
                child_process.wait()
            error_result_dict['error'] = err
            error_result_dict['time'] = exec_timeout
            return error_result_dict, err, False
        finally:
            if child_process and child_process.poll() is None:
                child_process.kill()
                child_process.wait()

    # Create temporary file
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as temp_file:
            temp_file.write(code)
            temp_file_path = temp_file.name
    except Exception as e:
        return False, f"Failed to create temp file: {e}", None

    # Test vectors
    test_vectors = [
        [],
        [1],
        [1, 2],
        [1, 2, 3],
        [3, 1, 2],
        [5, 2, 4, 1, 3],
        [48, 18, 44, 20, 16, 61, 26],
        [1, 2, 3, 4],
    ]
    
    # Add a few random vectors instead of many to speed up testing
    test_vectors.extend([
        [random.randint(0, 100) for _ in range(n)] for n in [5, 8, 12]
    ])

    # Run tests
    for vector in test_vectors:
        res_dict, err_msg, single_success = _run_single_test(vector, temp_file_path, exec_timeout)
        total_time += res_dict['time']
        test_results.append(res_dict)

        if not single_success:
            all_success = False
            failing_cases.append({
                'n': res_dict['n'], 'input': res_dict['input'], 'error': res_dict.get('error', 'Unknown failure'),
                'L_result': res_dict.get('L_result'), 'expected_L': res_dict.get('expected_L'),
                'R_result': res_dict.get('R_result'), 'expected_R': res_dict.get('expected_R'),
                'X_result': res_dict.get('X_result'), 'expected_X': res_dict.get('expected_X'),
            })

    # Cleanup
    if temp_file_path and os.path.exists(temp_file_path):
        try:
            os.unlink(temp_file_path)
        except:
            pass

    # Memory tracking (simplified)
    try:
        process = psutil.Process()
        max_memory_kb = process.memory_info().rss / 1024
    except:
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
        issue_str = json.dumps({'failing_cases': failing_cases[:3]}, ensure_ascii=False, indent=2)  # Limit output size
        return False, issue_str, summary

def llm_query(model: str, prompt: str, retries_config: Dict, config: Dict, progress_queue: queue.Queue, stage: str = None) -> Optional[str]:
    """Enhanced LLM query with better error handling and logging"""
    local.current_model = model
    local.current_queue = progress_queue
    local.current_data = {'tried': [], 'errors': {}, 'success': None, 'model': model}
    
    request_timeout = config['CONSTANTS']['REQUEST_TIMEOUT']

    for attempt in range(retries_config['max_retries'] + 1):
        try:
            progress_queue.put((model, 'log', f'Attempt {attempt + 1} for {stage}'))
            
            response = g4f.ChatCompletion.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                provider=Provider.AnyProvider,
                timeout=request_timeout
            )
            
            if response and response.strip():
                return response.strip()
                
        except ModelNotFoundError as e:
            progress_queue.put((model, 'log', f'ModelNotFoundError: {e}'))
            return None
        except Exception as e:
            progress_queue.put((model, 'log', f'Attempt {attempt + 1} failed: {e}'))
            if attempt < retries_config['max_retries']:
                sleep_time = retries_config['backoff_factor'] * (2 ** attempt)
                time.sleep(sleep_time)
            continue

    return None

def process_model(model: str, config: Dict, progress_queue: queue.Queue) -> Dict:
    """Enhanced model processing with better state management"""
    iterations = []
    current_code = None
    prev_code = None
    
    def update_progress(stage_name, current, total):
        progress_queue.put((model, 'status', f'Stage: {stage_name} ({current}/{total})'))
        progress_queue.put((model, 'progress', (current, total)))

    def run_test(code_to_test, stage_name):
        if not code_to_test or not code_to_test.strip():
            progress_queue.put((model, 'log', f'Test {stage_name}: Skipped (no code)'))
            return False, "No code to test", None
            
        progress_queue.put((model, 'log', f'Test {stage_name}: Running...'))
        success, issue, summary = test_code(code_to_test, config)
        if success:
            progress_queue.put((model, 'log', f'Test {stage_name}: SUCCESS'))
        else:
            progress_queue.put((model, 'log', f'Test {stage_name}: FAILED - {issue[:200]}'))
        return success, issue, summary

    def run_llm_query(prompt, stage_name, retries_key='FIX'):
        progress_queue.put((model, 'log', f'Stage: {stage_name}'))
        retries_cfg = config['RETRIES'][retries_key]
        
        response = llm_query(model, prompt, retries_cfg, config, progress_queue, stage_name)
        
        tried = local.current_data.get('tried', [])
        success_p = local.current_data.get('success', None)
        
        if response:
            cleaned = clean_code(response)
            progress_queue.put((model, 'log', f'Got response (cleaned {len(cleaned)} chars)'))
            return cleaned, None, tried, success_p
        else:
            error_msg = config['CONSTANTS']['ERROR_NO_RESPONSE']
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

    # Main processing flow
    progress_queue.put((model, 'log', f'=== STARTING PROCESSING: {model} ==='))
    
    total_stages = 1 + 1 + 1 + 1 + (config['CONSTANTS']['NUM_REFACTOR_LOOPS'] * 2) + 1
    current_stage = 0

    # 1. Initial generation
    current_stage += 1
    update_progress(config['STAGES']['INITIAL'], current_stage, total_stages)
    prompt = config['PROMPTS']['INITIAL']
    current_code, llm_error, tried, s_provider = run_llm_query(prompt, 'initial', 'INITIAL')
    add_iteration('initial', current_code, llm_error, None, tried, s_provider)
    
    if llm_error or not current_code:
        return {'model': model, 'iterations': iterations, 'final_code': None,
                'final_test': {'success': False, 'summary': None, 'issue': 'No initial response'}}

    # 2. Initial test & fix
    current_stage += 1
    update_progress(config['STAGES']['FIX_INITIAL'], current_stage, total_stages)
    success, issue, summary = run_test(current_code, 'initial_test')
    
    if not success:
        prompt = config['PROMPTS']['FIX'].format(code=current_code, error=issue)
        current_code, llm_error, tried, s_provider = run_llm_query(prompt, 'fix_initial')
        add_iteration('fix_initial', current_code, llm_error, summary, tried, s_provider)
        if llm_error:
            # If fix fails, we still continue but with the original code
            current_code = None

    # 3. First refactor
    if current_code:
        current_stage += 1
        update_progress(config['STAGES']['REFACTOR_FIRST'], current_stage, total_stages)
        prev_code = current_code
        prompt = config['PROMPTS']['REFACTOR_NO_PREV'].format(code=current_code)
        current_code, llm_error, tried, s_provider = run_llm_query(prompt, 'refactor_first', 'INITIAL')
        add_iteration('refactor_first', current_code, llm_error, None, tried, s_provider)
        
        if llm_error:
            current_code = prev_code  # Revert on error

    # 4. Test after first refactor
    if current_code:
        current_stage += 1
        update_progress(config['STAGES']['FIX_AFTER_REFACTOR'], current_stage, total_stages)
        success, issue, summary = run_test(current_code, 'post_refactor_test')
        
        if not success:
            prompt = config['PROMPTS']['FIX'].format(code=current_code, error=issue)
            current_code, llm_error, tried, s_provider = run_llm_query(prompt, 'fix_after_refactor')
            add_iteration('fix_after_refactor', current_code, llm_error, summary, tried, s_provider)

    # 5. Refactor loops (limited to prevent infinite processing)
    if current_code:
        for i in range(config['CONSTANTS']['NUM_REFACTOR_LOOPS']):
            # Refactor
            current_stage += 1
            stage_name = f"{config['STAGES']['REFACTOR']}_{i+1}"
            update_progress(stage_name, current_stage, total_stages)
            
            prompt = config['PROMPTS']['REFACTOR'].format(code=current_code, prev=prev_code)
            prev_code = current_code
            current_code, llm_error, tried, s_provider = run_llm_query(prompt, stage_name, 'INITIAL')
            add_iteration(stage_name, current_code, llm_error, None, tried, s_provider)
            
            if llm_error:
                current_code = prev_code
                break

            # Test after refactor
            current_stage += 1
            stage_name = f"{config['STAGES']['FIX_LOOP']}_{i+1}"
            update_progress(stage_name, current_stage, total_stages)
            
            success, issue, summary = run_test(current_code, f'loop_test_{i+1}')
            if not success:
                prompt = config['PROMPTS']['FIX'].format(code=current_code, error=issue)
                current_code, llm_error, tried, s_provider = run_llm_query(prompt, stage_name)
                add_iteration(stage_name, current_code, llm_error, summary, tried, s_provider)
                if llm_error:
                    break

    # 6. Final test
    current_stage += 1
    update_progress('final_test', current_stage, total_stages)
    success, issue, summary = run_test(current_code if current_code else "", 'final_test')
    
    add_iteration('final', current_code, None if success else issue, summary, [], None)

    if success:
        progress_queue.put((model, 'status', 'Success (final test)'))
    else:
        progress_queue.put((model, 'status', 'Failed (final test)'))

    return {
        'model': model,
        'iterations': iterations,
        'final_code': current_code,
        'final_test': {'success': success, 'summary': summary, 'issue': issue}
    }

def save_results(results, folder, filename):
    """Safe results saving with backup"""
    if not os.path.exists(folder):
        try:
            os.makedirs(folder)
        except OSError as e:
            print(f"Error creating folder {folder}: {e}", file=sys.stderr)
            return
    
    path = os.path.join(folder, filename)
    try:
        # Save with temporary file first to prevent corruption
        temp_path = path + '.tmp'
        with open(temp_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        os.rename(temp_path, path)
    except Exception as e:
        print(f"Error saving file {path}: {e}", file=sys.stderr)

def main():
    """Enhanced main function with better resource management"""
    print("Loading model list...")
    try:
        models = get_models_list(CONFIG)
        if not models:
            print("No models found. Check configuration.", file=sys.stderr)
            return
        print(f"Found {len(models)} unique models for testing.")
    except Exception as e:
        print(f"Failed to load model list: {e}", file=sys.stderr)
        traceback.print_exc()
        return

    # Create results folder
    intermediate_folder = CONFIG['CONSTANTS']['INTERMEDIATE_FOLDER']
    os.makedirs(intermediate_folder, exist_ok=True)

    progress_queue = queue.Queue()
    all_results = {}
    
    # Limit models for testing (adjust as needed)
    MAX_MODELS_TO_TEST = 20  # Reduced for faster testing
    models_to_test = models[:MAX_MODELS_TO_TEST] if MAX_MODELS_TO_TEST > 0 else models
    print(f"--- STARTING TESTING ({len(models_to_test)} models) ---")

    start_time_main = perf_counter()
    
    try:
        with ThreadPoolExecutor(max_workers=CONFIG['CONSTANTS']['MAX_WORKERS']) as executor:
            futures = {executor.submit(process_model, model, CONFIG, progress_queue): model 
                      for model in models_to_test}
            
            completed_count = 0
            total_count = len(futures)

            while completed_count < total_count:
                # Check completed futures
                done_futures = [f for f in futures if f.done()]
                
                for future in done_futures:
                    model = futures.pop(future)
                    completed_count += 1
                    
                    try:
                        result = future.result()
                        all_results[model] = result
                        
                        final_success = result.get('final_test', {}).get('success', False)
                        status = "SUCCESS" if final_success else "FAILED"
                        
                        # Save successful code
                        if final_success and result.get('final_code'):
                            code_filename = f"{model.replace('/', '_')}_final.py"
                            code_path = os.path.join(intermediate_folder, code_filename)
                            try:
                                with open(code_path, 'w', encoding='utf-8') as f:
                                    f.write(result['final_code'])
                            except Exception as e:
                                print(f"Error saving code for {model}: {e}", file=sys.stderr)
                        
                        print(f"--- ({completed_count}/{total_count}) COMPLETED: {model} [{status}] ---")
                        
                    except Exception as e:
                        print(f"--- ({completed_count}/{total_count}) CRITICAL ERROR: {model} ---")
                        tb_str = traceback.format_exc()
                        print(tb_str, file=sys.stderr)
                        all_results[model] = {
                            'error': str(e), 
                            'traceback': tb_str, 
                            'iterations': [], 
                            'final_code': None, 
                            'final_test': {'success': False, 'summary': None, 'issue': str(e)}
                        }
                    
                    # Periodic saving
                    if completed_count % CONFIG['CONSTANTS']['N_SAVE'] == 0 or completed_count == total_count:
                        save_results(all_results, intermediate_folder, f"intermediate_results_{completed_count}.json")
                
                # Process progress queue
                try:
                    while not progress_queue.empty():
                        model, msg_type, message = progress_queue.get_nowait()
                        # Optional: Uncomment for detailed logging
                        # print(f"{msg_type.upper()} [{model}]: {message}")
                except queue.Empty:
                    pass
                
                time.sleep(0.1)
                
    except KeyboardInterrupt:
        print("\nStopping on user request...")
    except Exception as e:
        print(f"Unexpected error in main: {e}")
        traceback.print_exc()
    finally:
        end_time_main = perf_counter()
        total_time_main = end_time_main - start_time_main
        
        # Final save
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        final_filename = f"final_results_{timestamp}.json"
        save_results(all_results, intermediate_folder, final_filename)
        
        # Also save as latest for easy access
        save_results(all_results, intermediate_folder, "final_results_latest.json")
        
        print("--- TESTING COMPLETED ---")
        print(f"Total execution time: {total_time_main:.2f} seconds")
        print(f"Final results saved in: {os.path.join(intermediate_folder, final_filename)}")
        
        # Summary
        success_count = sum(1 for res in all_results.values() if res.get('final_test', {}).get('success', False))
        fail_count = len(all_results) - success_count
        print(f"Summary: {success_count} Successful, {fail_count} Failed")

if __name__ == "__main__":
    main()
import json
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
from collections import Counter

# --- Settings ---
# *** FIX: Translated to English ***
RESULTS_FOLDER = 'results'
REPORT_FILENAME = 'benchmark_analysis_report.txt'
PIE_CHART_FILENAME = 'benchmark_pie_chart.png'
BAR_CHART_FILENAME = 'benchmark_top5_performance.png'
TOP_N_MODELS = 5
# ---

def find_latest_results_file(folder: str) -> str | None:
    """Finds the latest results JSON file in the folder."""
    # *** FIX: Check in the correct folder ***
    if not os.path.isdir(folder):
        print(f"Error: Results folder '{folder}' not found.", file=sys.stderr)
        print("Please run 'test_code_LRX.py' first to generate results.", file=sys.stderr)
        return None
        
    list_of_files = glob.glob(os.path.join(folder, 'final_results_*.json'))
    if not list_of_files:
        print(f"No 'final_results_*.json' files found in '{folder}'.", file=sys.stderr)
        return None
    latest_file = max(list_of_files, key=os.path.getctime)
    print(f"Found results file: {latest_file}")
    return latest_file

def load_and_parse_data(filepath: str) -> pd.DataFrame:
    """
    Loads the JSON file and parses it into a DataFrame.
    JSON structure: { 'model_name': {'final_test': {...}, ...} }
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error: Failed to read or decode file {filepath}. {e}")
        return pd.DataFrame()

    parsed_data = []
    for model_name, results in data.items():
        final_test = results.get('final_test', {})
        success = final_test.get('success', False)
        summary = final_test.get('summary', {})
        
        if success:
            # For successful models, get real metrics
            avg_time = summary.get('avg_time', float('inf'))
            max_mem = summary.get('max_memory_kb')
            issue = "Success"
        else:
            # For failed models, set 'inf' and get the reason
            avg_time = float('inf')
            max_mem = None
            issue = final_test.get('issue', 'Unknown Error')
            
            # Simplify long errors (often JSON from tests)
            if isinstance(issue, str) and issue.startswith('{'):
                try:
                    # Try to get the first failure reason
                    issue_json = json.loads(issue)
                    first_fail = issue_json.get('failing_cases', [{}])[0]
                    issue = first_fail.get('error', 'Test mismatch')
                except json.JSONDecodeError:
                    issue = "Test mismatch (JSON output)" # JSON parse error

        parsed_data.append({
            'Model': model_name,
            'Success': success,
            'Avg_Time (s)': avg_time,
            'Max_Memory (KB)': max_mem,
            'Result_Details': issue
        })
        
    return pd.DataFrame(parsed_data)

def generate_report(df: pd.DataFrame):
    """
    Generates a text report, tables, and saves plots.
    """
    if df.empty:
        print("No data to analyze.")
        return

    # --- 1. Prepare data ---
    total_models = len(df)
    successful_df = df[df['Success'] == True].sort_values(by='Avg_Time (s)')
    failed_df = df[df['Success'] == False]
    
    success_count = len(successful_df)
    fail_count = len(failed_df)
    
    # Error statistics
    error_counts = Counter(failed_df['Result_Details'])
    
    # --- 2. Generate text report (to file and console) ---
    # *** FIX: Translated report text ***
    report_lines = []
    report_lines.append("="*80)
    report_lines.append("                LRX ALGORITHM IMPLEMENTATION BENCHMARK REPORT")
    report_lines.append("="*80)
    report_lines.append("\n## 1. Ranking Criteria\n")
    report_lines.append("The best model is determined by the following priority criteria:")
    report_lines.append("  1. **Correctness (Success = True):** The most important criterion. The model must")
    report_lines.append("     pass *all* tests, including edge cases (empty lists, n=1, n=2).")
    report_lines.append("  2. **Average Execution Time (Avg_Time):** Among *correct* models,")
    report_lines.append("     preference is given to those whose code runs faster.")
    report_lines.append("  3. **Memory Usage (Max_Memory):** An additional criterion for evaluating")
    report_lines.append("     efficiency (less indicative than time in this test).\n")

    report_lines.append("---")
    report_lines.append("## 2. Overall Statistics\n")
    
    summary_table = pd.DataFrame({
        'Metric': ['Total models tested', 'Successfully passed tests', 'Failed tests'],
        'Value': [total_models, success_count, fail_count]
    })
    report_lines.append(summary_table.to_string(index=False))
    report_lines.append("\n")

    # --- 3. Best Model Selection ---
    report_lines.append("---")
    report_lines.append("## 3. Best Model Selection\n")
    if not successful_df.empty:
        best_model = successful_df.iloc[0]
        report_lines.append(f"🏆 **Best Model: {best_model['Model']}** 🏆\n")
        report_lines.append("It showed the best average execution time among all correct implementations.\n")
        report_lines.append(f"  - Average Time: {best_model['Avg_Time (s)']:.6f} sec.")
        report_lines.append(f"  - Max Memory: {best_model['Max_Memory (KB)']} KB\n")
    else:
        report_lines.append("❌ **Best Model Not Found.**\n")
        report_lines.append("None of the tested models were able to pass the full test suite.\n")

    # --- 4. Top N Successful Models (Table) ---
    if not successful_df.empty:
        report_lines.append("---")
        report_lines.append(f"## 4. Top {TOP_N_MODELS} Successful Models (by Execution Time)\n")
        
        # Format for output
        top_n_df = successful_df.head(TOP_N_MODELS).copy()
        top_n_df['Avg_Time (s)'] = top_n_df['Avg_Time (s)'].map('{:,.6f}'.format)
        top_n_df['Max_Memory (KB)'] = top_n_df['Max_Memory (KB)'].map('{:,.0f}'.format)
        top_n_df.drop(columns=['Success', 'Result_Details'], inplace=True)
        top_n_df.reset_index(drop=True, inplace=True)
        top_n_df.index = top_n_df.index + 1
        top_n_df.index.name = "Rank"
        
        report_lines.append(top_n_df.to_string())
        report_lines.append("\n")
    
    # --- 5. Failure Analysis (Table) ---
    if not failed_df.empty:
        report_lines.append("---")
        report_lines.append("## 5. Analysis of Main Failure Reasons\n")
        
        error_df = pd.DataFrame(error_counts.items(), columns=['Failure Reason', 'Model Count'])
        error_df = error_df.sort_values(by='Model Count', ascending=False)
        error_df.reset_index(drop=True, inplace=True)
        
        report_lines.append(error_df.to_string(index=False))
        report_lines.append("\n")

    report_lines.append("="*80)
    report_lines.append(f"Full report saved to: {REPORT_FILENAME}")
    report_lines.append(f"Charts saved to: {PIE_CHART_FILENAME}, {BAR_CHART_FILENAME}")
    report_lines.append("="*80)

    # --- 6. Print to console and save to file ---
    report_text = "\n".join(report_lines)
    print(report_text)
    
    try:
        with open(REPORT_FILENAME, 'w', encoding='utf-8') as f:
            f.write(report_text)
    except Exception as e:
        print(f"Failed to save text report: {e}")

    # --- 7. Generate Plots ---
    # *** FIX: Completed the plotting logic ***
    
    # Plot 1: Pie Chart (Success vs Fail)
    try:
        plt.figure(figsize=(8, 6))
        labels = ['Successful', 'Failed']
        sizes = [success_count, fail_count]
        colors = ['#4CAF50', '#F44336'] # Green, Red
        
        if success_count == 0 and fail_count == 0:
             print("No data to plot.")
             return
        elif success_count == 0:
             labels = ['Failed']
             sizes = [fail_count]
             colors = ['#F44336']
        elif fail_count == 0:
             labels = ['Successful']
             sizes = [success_count]
             colors = ['#4CAF50']

        plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        plt.title('Benchmark Results: Success vs. Failure')
        plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        plt.savefig(PIE_CHART_FILENAME)
        plt.close()
        print(f"Pie chart saved to {PIE_CHART_FILENAME}")
    except Exception as e:
        print(f"Failed to generate pie chart: {e}")

    # Plot 2: Bar Chart (Top N Performance)
    if not successful_df.empty:
        try:
            # Sort ascending for time (lower is better), then reverse for plotting
            top_n_plot_df = successful_df.head(TOP_N_MODELS).sort_values(by='Avg_Time (s)', ascending=False)
            
            plt.figure(figsize=(10, 6))
            plt.barh(top_n_plot_df['Model'], top_n_plot_df['Avg_Time (s)'], color='skyblue')
            plt.xlabel('Average Execution Time (s)')
            plt.ylabel('Model')
            plt.title(f'Top {TOP_N_MODELS} Successful Models by Performance (Lower is Better)')
            plt.tight_layout()
            plt.savefig(BAR_CHART_FILENAME)
            plt.close()
            print(f"Bar chart saved to {BAR_CHART_FILENAME}")
        except Exception as e:
            print(f"Failed to generate bar chart: {e}")
    else:
        print("Skipping performance bar chart: no successful models.")

def main():
    """Main function to find, load, parse, and report data."""
    # *** FIX: Use English folder name ***
    latest_file = find_latest_results_file(RESULTS_FOLDER)
    
    if latest_file:
        df = load_and_parse_data(latest_file)
        if not df.empty:
            generate_report(df)
        else:
            print("Failed to load or parse data.")
    else:
        print("No results file found. Exiting analysis.")

if __name__ == "__main__":
    main()
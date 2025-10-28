import json
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
from collections import Counter
import sys

# --- Settings ---
RESULTS_FOLDER = 'промежуточные результаты'
REPORT_FILENAME = 'benchmark_analysis_report.txt'
PIE_CHART_FILENAME = 'benchmark_pie_chart.png'
BAR_CHART_FILENAME = 'benchmark_top5_performance.png'
TOP_N_MODELS = 5
# ---

def find_latest_results_file(folder: str) -> str | None:
    """Finds the most recent results file with multiple fallback options"""
    # Try multiple file patterns
    patterns = [
        'final_results_latest.json',  # Primary fallback
        'final_results_*.json',       # Timestamped files
        'intermediate_results_*.json' # Intermediate files
    ]
    
    for pattern in patterns:
        list_of_files = glob.glob(os.path.join(folder, pattern))
        if list_of_files:
            # Sort by creation time and get the latest
            latest_file = max(list_of_files, key=os.path.getctime)
            print(f"Found results file: {latest_file}")
            return latest_file
    
    return None

def load_and_parse_data(filepath: str) -> pd.DataFrame:
    """
    Loads and parses JSON results into DataFrame with enhanced error handling
    """
    if not os.path.exists(filepath):
        print(f"Error: File {filepath} does not exist")
        return pd.DataFrame()
        
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error: Failed to read or decode file {filepath}. {e}")
        return pd.DataFrame()

    parsed_data = []
    for model_name, results in data.items():
        # Handle different result structures
        if isinstance(results, dict):
            final_test = results.get('final_test', {})
            success = final_test.get('success', False)
            summary = final_test.get('summary', {})
            
            if success:
                avg_time = summary.get('avg_time', float('inf'))
                max_mem = summary.get('max_memory_kb')
                issue = "Success"
            else:
                avg_time = float('inf')
                max_mem = None
                issue = final_test.get('issue', 'Unknown Error')
                
                # Simplify error messages
                if isinstance(issue, str):
                    if issue.startswith('{'):
                        try:
                            issue_json = json.loads(issue)
                            first_fail = issue_json.get('failing_cases', [{}])[0]
                            issue = first_fail.get('error', 'Test mismatch')[:100]  # Limit length
                        except json.JSONDecodeError:
                            issue = "Test mismatch (JSON output)"
                    else:
                        # Truncate long error messages
                        issue = issue[:150] + "..." if len(issue) > 150 else issue
        else:
            # Handle invalid result format
            success = False
            avg_time = float('inf')
            max_mem = None
            issue = "Invalid result format"

        parsed_data.append({
            'Model': model_name,
            'Success': success,
            'Avg_Time (s)': avg_time,
            'Max_Memory (KB)': max_mem,
            'Result_Details': issue
        })
        
    df = pd.DataFrame(parsed_data)
    print(f"Loaded data for {len(df)} models")
    return df

def generate_report(df: pd.DataFrame):
    """
    Generates comprehensive analysis report with charts
    """
    if df.empty:
        print("No data available for analysis.")
        return

    # --- 1. Data Preparation ---
    total_models = len(df)
    successful_df = df[df['Success'] == True].sort_values(by='Avg_Time (s)')
    failed_df = df[df['Success'] == False]
    
    success_count = len(successful_df)
    fail_count = len(failed_df)
    
    error_counts = Counter(failed_df['Result_Details'])
    
    # --- 2. Generate Text Report ---
    report_lines = []
    report_lines.append("="*80)
    report_lines.append("           BENCHMARK REPORT: LRX ALGORITHM IMPLEMENTATIONS")
    report_lines.append("="*80)
    report_lines.append("\n## 1. Ranking Criteria\n")
    report_lines.append("Best models are selected based on:")
    report_lines.append("  1. **Correctness (Success = True):** Must pass all test cases")
    report_lines.append("  2. **Average Execution Time:** Faster is better among correct implementations")
    report_lines.append("  3. **Memory Usage:** Lower memory consumption is preferred\n")

    report_lines.append("---")
    report_lines.append("## 2. Overall Statistics\n")
    
    summary_table = pd.DataFrame({
        'Metric': ['Total Models Tested', 'Successfully Passed', 'Failed'],
        'Count': [total_models, success_count, fail_count],
        'Percentage': [
            '100%', 
            f'{(success_count/total_models*100):.1f}%' if total_models > 0 else '0%',
            f'{(fail_count/total_models*100):.1f}%' if total_models > 0 else '0%'
        ]
    })
    report_lines.append(summary_table.to_string(index=False))
    report_lines.append("\n")

    # --- 3. Best Model Selection ---
    report_lines.append("---")
    report_lines.append("## 3. Best Model Selection\n")
    if not successful_df.empty:
        best_model = successful_df.iloc[0]
        report_lines.append(f"🏆 **Best Model: {best_model['Model']}** 🏆\n")
        report_lines.append("This model achieved the best performance among all correct implementations.\n")
        report_lines.append(f"  • Average Time: {best_model['Avg_Time (s)']:.6f} seconds")
        if best_model['Max_Memory (KB)']:
            report_lines.append(f"  • Max Memory: {best_model['Max_Memory (KB)']:,.0f} KB")
        report_lines.append("")
    else:
        report_lines.append("❌ **No successful models found.**\n")
        report_lines.append("None of the tested models passed all test cases.\n")

    # --- 4. Top N Successful Models ---
    if not successful_df.empty:
        report_lines.append("---")
        report_lines.append(f"## 4. Top-{TOP_N_MODELS} Successful Models (by execution time)\n")
        
        top_n_df = successful_df.head(TOP_N_MODELS).copy()
        top_n_df['Avg_Time (s)'] = top_n_df['Avg_Time (s)'].map('{:,.6f}'.format)
        if 'Max_Memory (KB)' in top_n_df.columns:
            top_n_df['Max_Memory (KB)'] = top_n_df['Max_Memory (KB)'].map(lambda x: '{:,.0f}'.format(x) if pd.notna(x) else 'N/A')
        top_n_df = top_n_df[['Model', 'Avg_Time (s)', 'Max_Memory (KB)'] if 'Max_Memory (KB)' in top_n_df.columns else ['Model', 'Avg_Time (s)']]
        top_n_df.reset_index(drop=True, inplace=True)
        top_n_df.index = top_n_df.index + 1
        top_n_df.index.name = "Rank"
        
        report_lines.append(top_n_df.to_string())
        report_lines.append("\n")
    
    # --- 5. Error Analysis ---
    if not failed_df.empty:
        report_lines.append("---")
        report_lines.append("## 5. Failure Analysis\n")
        
        error_df = pd.DataFrame(error_counts.most_common(10), columns=['Failure Reason', 'Model Count'])
        error_df['Percentage'] = (error_df['Model Count'] / fail_count * 100).map('{:.1f}%'.format)
        error_df.reset_index(drop=True, inplace=True)
        
        report_lines.append(error_df.to_string(index=False))
        report_lines.append("\n")

    report_lines.append("="*80)
    report_lines.append(f"Full report saved to: {REPORT_FILENAME}")
    report_lines.append(f"Charts saved to: {PIE_CHART_FILENAME}, {BAR_CHART_FILENAME}")
    report_lines.append("="*80)

    # --- 6. Save and Display Report ---
    report_text = "\n".join(report_lines)
    print(report_text)
    
    try:
        with open(REPORT_FILENAME, 'w', encoding='utf-8') as f:
            f.write(report_text)
        print(f"Report saved to {REPORT_FILENAME}")
    except Exception as e:
        print(f"Failed to save text report: {e}")

    # --- 7. Generate Charts ---
    
    # Chart 1: Pie chart (Success vs Fail)
    try:
        plt.figure(figsize=(10, 8))
        
        if success_count > 0 or fail_count > 0:
            labels = ['Success', 'Failure']
            sizes = [success_count, fail_count]
            colors = ['#4CAF50', '#F44336']
            explode = (0.1, 0)  # explode the successful slice
            
            plt.pie(sizes, explode=explode, labels=labels, colors=colors, 
                   autopct='%1.1f%%', shadow=True, startangle=90,
                   textprops={'fontsize': 12, 'fontweight': 'bold'})
            
            plt.title(f'Model Success Rate (Total: {total_models} models)', 
                     fontsize=16, fontweight='bold', pad=20)
            plt.axis('equal')
            plt.tight_layout()
            plt.savefig(PIE_CHART_FILENAME, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Pie chart saved to {PIE_CHART_FILENAME}")
        else:
            print("No data available for pie chart")
            
    except Exception as e:
        print(f"Failed to create pie chart: {e}")

    # Chart 2: Top-N Models Bar chart
    try:
        if not successful_df.empty and len(successful_df) > 0:
            top_n_chart_df = successful_df.head(TOP_N_MODELS).sort_values(by='Avg_Time (s)', ascending=True)
            
            plt.figure(figsize=(12, max(6, TOP_N_MODELS * 0.6)))
            bars = plt.barh(range(len(top_n_chart_df)), top_n_chart_df['Avg_Time (s)'], 
                           color='skyblue', edgecolor='navy', alpha=0.7)
            
            plt.xlabel('Average Execution Time (seconds)', fontsize=12, fontweight='bold')
            plt.ylabel('Model', fontsize=12, fontweight='bold')
            plt.title(f'Top-{TOP_N_MODELS} Fastest Correct Models', fontsize=14, fontweight='bold')
            plt.yticks(range(len(top_n_chart_df)), top_n_chart_df['Model'])
            
            # Add value labels on bars
            for i, bar in enumerate(bars):
                width = bar.get_width()
                plt.text(width, bar.get_y() + bar.get_height()/2, 
                        f'{width:.6f}s', 
                        ha='left', va='center', fontweight='bold')
            
            plt.grid(axis='x', linestyle='--', alpha=0.6)
            plt.tight_layout()
            plt.savefig(BAR_CHART_FILENAME, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Bar chart saved to {BAR_CHART_FILENAME}")
        else:
            print("No successful models for bar chart")
            
    except Exception as e:
        print(f"Failed to create performance bar chart: {e}")

def main():
    """Main function with enhanced error handling and user feedback"""
    print(f"Searching for results files in '{RESULTS_FOLDER}'...")
    
    if not os.path.exists(RESULTS_FOLDER):
        print(f"Error: Results folder '{RESULTS_FOLDER}' does not exist.")
        print("Please run 'test_code_LRX.py' first to generate results.")
        return

    latest_file = find_latest_results_file(RESULTS_FOLDER)
    
    if not latest_file:
        print(f"Error: No results files found in '{RESULTS_FOLDER}'.")
        print("Please run 'test_code_LRX.py' first to generate results.")
        return

    print(f"Loading data from: {latest_file}")
    df = load_and_parse_data(latest_file)
    
    if df.empty:
        print("No valid data loaded. Analysis aborted.")
        return
        
    print("Generating report and charts...")
    generate_report(df)
    print("Analysis completed successfully.")

if __name__ == "__main__":
    main()
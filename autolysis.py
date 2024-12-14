import os
import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import openai
from tabulate import tabulate
from tenacity import retry, stop_after_attempt, wait_fixed
import signal

# Ensure seaborn is installed
def ensure_dependencies():
    try:
        import seaborn
    except ImportError:
        print("Seaborn is not installed. Please install it using 'pip install seaborn'.")
        sys.exit(1)

ensure_dependencies()

# Set API token securely
def set_openai_api():
    openai.api_key = os.getenv("AI_PROXY")
    if not openai.api_key:
        raise EnvironmentError("Please set the AI_PROXY environment variable with your OpenAI API key.")

set_openai_api()

# Timeout exception class
class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException("Operation timed out!")

signal.signal(signal.SIGALRM, timeout_handler)

# Dynamic prompt generation
def generate_llm_prompt(data_summary, analysis_results):
    return (
        f"Analyze the following dataset summary and results:\n"
        f"\nDataset Summary:\n{data_summary}\n"
        f"\nAnalysis Results:\n{analysis_results}\n"
        f"\nProvide insights, implications, and suggest further analysis."
    )

# Analyze dataset
def analyze_dataset(file_path):
    try:
        data = pd.read_csv(file_path)
        summary = data.describe(include='all').transpose()
        null_values = data.isnull().sum()
        
        # Correlation heatmap
        plt.figure(figsize=(10, 6))
        correlation = data.corr()
        sns.heatmap(correlation, annot=True, cmap="coolwarm", fmt=".2f")
        plt.title("Correlation Heatmap")
        heatmap_path = f"{os.path.splitext(file_path)[0]}_correlation.png"
        plt.savefig(heatmap_path)
        plt.close()

        # Outlier detection
        for col in data.select_dtypes(include=['float64', 'int64']).columns:
            plt.figure(figsize=(10, 4))
            sns.boxplot(x=data[col])
            plt.title(f"Outliers in {col}")
            boxplot_path = f"{os.path.splitext(file_path)[0]}_{col}_boxplot.png"
            plt.savefig(boxplot_path)
            plt.close()

        return data, summary, null_values, heatmap_path

    except Exception as e:
        print(f"Error during dataset analysis: {e}")
        sys.exit(1)

# Generate README
def generate_readme(file_path, summary, null_values, insights):
    readme_content = (
        f"# Analysis Report for {file_path}\n\n"
        f"## Dataset Summary\n"
        f"{tabulate(summary, headers='keys', tablefmt='github')}\n\n"
        f"## Null Values\n"
        f"{tabulate(null_values.reset_index(), headers=['Column', 'Null Values'], tablefmt='github')}\n\n"
        f"## Insights and Implications\n"
        f"{insights}\n"
    )

    readme_path = f"{os.path.splitext(file_path)[0]}_README.md"
    with open(readme_path, "w") as f:
        f.write(readme_content)

    print(f"README generated at {readme_path}")

# Interact with LLM with retry mechanism
@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def interact_with_llm(prompt):
    try:
        signal.alarm(120)  # Set timeout for LLM interaction
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            max_tokens=500
        )
        signal.alarm(0)  # Disable timeout after success
        return response.choices[0].text.strip()
    except TimeoutException:
        print("LLM interaction timed out!")
        sys.exit(1)
    except Exception as e:
        print(f"Error interacting with LLM: {e}")
        raise

# Main function
def main():
    if len(sys.argv) != 2:
        print("Usage: python autolysis.py <dataset.csv>")
        sys.exit(1)

    file_path = sys.argv[1]

    data, summary, null_values, heatmap_path = analyze_dataset(file_path)
    
    # Dynamic prompt generation and interaction
    data_summary = summary.to_string()
    null_summary = null_values.to_string()
    prompt = generate_llm_prompt(data_summary, null_summary)
    
    try:
        insights = interact_with_llm(prompt)
    except Exception as e:
        print(f"Failed to get insights from LLM: {e}")
        sys.exit(1)

    # Generate README
    generate_readme(file_path, summary, null_values, insights)

if __name__ == "__main__":
    main()  

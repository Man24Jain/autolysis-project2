# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "seaborn",
#   "pandas",
#   "matplotlib",
#   "openai",
#   "tabulate",
#   "tenacity",
#   "numpy",
# ]
# ///

import os
import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import openai
from tabulate import tabulate
from tenacity import retry, stop_after_attempt, wait_fixed
import signal
import subprocess

# Ensure dependencies are installed
def ensure_dependencies():
    dependencies = ["seaborn", "pandas", "matplotlib", "openai", "tabulate", "tenacity", "numpy"]
    for dep in dependencies:
        try:
            __import__(dep)
        except ImportError:
            print(f"{dep} is not installed. Installing now...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", dep])

ensure_dependencies()

# Set OpenAI API key
def set_openai_api():
    openai.api_key = os.getenv("AIPROXY_TOKEN")
    if not openai.api_key:
        raise EnvironmentError("Please set the AIPROXY_TOKEN environment variable.")

set_openai_api()

# Timeout exception class
class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException("Operation timed out!")

signal.signal(signal.SIGALRM, timeout_handler)

# Generate LLM prompt
def generate_llm_prompt(data_summary, null_values):
    return (
        f"You are an AI data storyteller. Analyze the dataset below and provide insights in a "
        f"narrative format that includes:\n"
        f"1. Patterns and anomalies in the data.\n"
        f"2. Key statistical findings presented clearly.\n"
        f"3. Recommendations for decision-makers.\n\n"
        f"Dataset Summary:\n{data_summary}\n\n"
        f"Null Values Summary:\n{null_values}\n\n"
        f"Ensure the output is engaging, structured, and actionable."
    )

# Analyze dataset
def analyze_dataset(file_path):
    try:
        data = pd.read_csv(file_path)
        summary = data.describe(include='all').transpose()
        null_values = data.isnull().sum()

        # Generate correlation heatmap
        plt.figure(figsize=(10, 6))
        correlation = data.corr()
        sns.heatmap(correlation, annot=True, cmap="coolwarm", fmt=".2f")
        plt.title("Correlation Heatmap")
        heatmap_path = f"{os.path.splitext(file_path)[0]}_correlation.png"
        plt.savefig(heatmap_path)
        plt.close()

        # Generate boxplots for numeric columns
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
        f"## Insights and Recommendations\n"
        f"{insights}\n"
    )

    readme_path = f"{os.path.splitext(file_path)[0]}_README.md"
    with open(readme_path, "w") as f:
        f.write(readme_content)

    print(f"README generated at {readme_path}")

# Interact with LLM
@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def interact_with_llm(prompt):
    try:
        signal.alarm(120)  # Set timeout for LLM interaction
        response = openai.Completion.create(
            engine="gpt-4o-mini",
            prompt=prompt,
            max_tokens=2000,  # Increased token limit
            temperature=0.7,
            n=1
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

    # Generate prompt and interact with LLM
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

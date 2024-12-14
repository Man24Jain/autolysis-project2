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
import random
import string

# Ensure seaborn is installed
def ensure_dependencies():
    try:
        import seaborn
    except ImportError:
        print("Seaborn is not installed. Installing now...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "seaborn"])

ensure_dependencies()

# Set API token securely
def set_openai_api():
    openai.api_key = os.getenv("AIPROXY_TOKEN")
    if not openai.api_key:
        raise EnvironmentError("Please set the AIPROXY_TOKEN environment variable with your OpenAI API key.")

set_openai_api()

# Timeout exception class
class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException("Operation timed out!")

signal.signal(signal.SIGALRM, timeout_handler)

# Dynamic prompt generation with robust instructions
def generate_llm_prompt(data_summary, null_values, inject_prompt=False):
    base_prompt = (
        f"You are a data analyst. Analyze the dataset described below and provide insights. "
        f"The response should include:\n"
        f"1. Patterns or anomalies in the data.\n"
        f"2. Key statistical findings.\n"
        f"3. Implications of these findings.\n\n"
        f"Dataset Summary:\n{data_summary}\n\n"
        f"Null Values Summary:\n{null_values}\n\n"
        f"Ensure insights are clear, concise, and actionable."
    )
    if inject_prompt:
        base_prompt += "\n\nFor extra creativity, imagine this dataset represents a narrative from the future."
    return base_prompt

# Preprocess dataset for unique column naming and shuffling
def preprocess_data(data):
    columns = list(data.columns)
    random.shuffle(columns)  # Shuffle column order for diversity
    data = data[columns]
    unique_columns = [f"{random.choice(string.ascii_uppercase)}_{col}" for col in columns]
    data.columns = unique_columns
    return data

# Analyze dataset with reproducibility
def analyze_dataset(file_path):
    try:
        data = pd.read_csv(file_path)

        # Preprocess for uniqueness
        data = preprocess_data(data)

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

        # Additional visualization: Pairplot
        try:
            sns.pairplot(data, diag_kind='kde')
            pairplot_path = f"{os.path.splitext(file_path)[0]}_pairplot.png"
            plt.savefig(pairplot_path)
            plt.close()
            print(f"Pairplot saved at {pairplot_path}")
        except Exception as e:
            print(f"Error generating pairplot: {e}")

        return data, summary, null_values, heatmap_path

    except Exception as e:
        print(f"Error during dataset analysis: {e}")
        sys.exit(1)

# Generate README with storytelling elements
def generate_readme(file_path, summary, null_values, insights):
    readme_content = (
        f"# Analysis Report for {file_path}\n\n"
        f"Once upon a time, in the world of datasets, we encountered {file_path}.\n"
        f"This dataset told a story filled with patterns, anomalies, and insightful discoveries.\n\n"
        f"## Dataset Summary\n"
        f"Here are the key statistics that lay the foundation of this tale:\n"
        f"{tabulate(summary, headers='keys', tablefmt='github')}\n\n"
        f"## Null Values\n"
        f"Every data story has its gaps. Here's what we found missing:\n"
        f"{tabulate(null_values.reset_index(), headers=['Column', 'Null Values'], tablefmt='github')}\n\n"
        f"## Insights and Implications\n"
        f"The twists and turns of the analysis revealed the following:\n"
        f"{insights}\n\n"
        f"### Bonus Visuals\n"
        f"We also uncovered some hidden relationships—check out the visuals saved alongside!"
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
            engine="gpt-4o-mini",
            prompt=prompt,
            max_tokens=1500,  # Increased token limit for thorough analysis
            temperature=0.7,  # Added variability for richer responses
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
        print("Usage: uv run autolysis.py <dataset.csv>")
        sys.exit(1)

    file_path = sys.argv[1]

    data, summary, null_values, heatmap_path = analyze_dataset(file_path)

    # Dynamic prompt generation and interaction
    data_summary = summary.to_string()
    null_summary = null_values.to_string()
    prompt = generate_llm_prompt(data_summary, null_summary, inject_prompt=True)

    try:
        insights = interact_with_llm(prompt)
    except Exception as e:
        print(f"Failed to get insights from LLM: {e}")
        sys.exit(1)

    # Generate README
    generate_readme(file_path, summary, null_values, insights)

if __name__ == "__main__":
    main()

import os
import pandas as pd
import matplotlib.pyplot as plt
from tenacity import retry, stop_after_attempt, wait_fixed
import openai

# Set up OpenAI API token
AIPROXY_TOKEN = os.getenv("AIPROXY_TOKEN")
openai.api_key = AIPROXY_TOKEN

# Helper: Retry logic for LLM calls
@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def call_llm(prompt, model="gpt-4o-mini", functions=None):
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            functions=functions or []
        )
        return response["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"LLM call failed: {e}")
        raise

# Load and preprocess data
def load_and_clean_data(filename):
    try:
        df = pd.read_csv(filename)
        print(f"Data loaded: {filename}, shape: {df.shape}")
        return df
    except Exception as e:
        print(f"Error loading {filename}: {e}")
        raise

# Perform data analysis
def analyze_data(df):
    summary = df.describe(include='all').to_dict()
    print(f"Analysis summary generated: {summary.keys()}")
    return summary

# Visualize data
def visualize_data(df, output_dir="charts"):
    os.makedirs(output_dir, exist_ok=True)
    try:
        for column in df.select_dtypes(include='number').columns:
            plt.figure()
            df[column].plot(kind="hist", title=column)
            output_path = os.path.join(output_dir, f"{column}.png")
            plt.savefig(output_path)
            plt.close()
            print(f"Saved chart: {output_path}")
    except Exception as e:
        print(f"Error creating visualizations: {e}")
        raise

# Generate README.md
def generate_readme(analysis_summary, output_dir="charts"):
    markdown = "# Automated Data Analysis\n\n"
    markdown += "## Data Summary\n\n"
    for key, stats in analysis_summary.items():
        markdown += f"- **{key}**: {stats}\n"

    markdown += "\n## Visualizations\n\n"
    for chart in os.listdir(output_dir):
        if chart.endswith(".png"):
            markdown += f"![{chart}](./{output_dir}/{chart})\n"

    with open("README.md", "w") as f:
        f.write(markdown)
    print("README.md generated")

# Main function
def main(filename):
    try:
        df = load_and_clean_data(filename)
        analysis_summary = analyze_data(df)
        visualize_data(df)
        generate_readme(analysis_summary)
        print("Process completed successfully!")
    except Exception as e:
        print(f"Error during process: {e}")

# Entry point
if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: uv run autolysis.py <dataset.csv>")
        sys.exit(1)
    main(sys.argv[1])

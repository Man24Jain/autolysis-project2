import os
import sys
import random
import httpx
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import chardet
from pathlib import Path

# Set the Agg backend for matplotlib (non-interactive environments)
import matplotlib
matplotlib.use('Agg')

# Function to detect file encoding
def detect_encoding(file_path):
    with open(file_path, 'rb') as f:
        result = chardet.detect(f.read(10000))
    return result['encoding']

# Function to load CSV file
def load_csv(file_path):
    try:
        encoding = detect_encoding(file_path)
        data = pd.read_csv(file_path, encoding=encoding)
        return data
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        sys.exit(1)

# Function to summarize the dataset
def summarize_dataset(data):
    try:
        summary = {
            "columns": list(data.columns),
            "dtypes": data.dtypes.to_dict(),
            "missing_values": data.isnull().sum().to_dict(),
            "describe": data.describe(include='all').to_dict()
        }
        return summary
    except Exception as e:
        print(f"Error summarizing dataset: {e}")
        sys.exit(1)

# Function to generate visualizations
def generate_visualizations(data, output_dir):
    numeric_columns = data.select_dtypes(include=['number']).columns
    if numeric_columns.empty:
        print("No numeric columns found for visualization.")
        return

    os.makedirs(output_dir, exist_ok=True)

    for col in numeric_columns:
        try:
            plt.figure(figsize=(8, 6))
            sns.histplot(data[col], kde=True, color=random.choice(sns.color_palette()))
            plt.title(f"Distribution of {col}")
            plt.xlabel(col)
            plt.ylabel("Frequency")
            plt.savefig(os.path.join(output_dir, f"{col}_distribution.png"))
            plt.close()
        except Exception as e:
            print(f"Error generating visualization for {col}: {e}")

# Function to call GPT-4o-Mini API for narrative generation
def generate_narrative(summary):
    try:
        api_url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {os.getenv('AIPROXY_TOKEN')}"
        }
        payload = {
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": "You are a helpful data analysis assistant."},
                {"role": "user", "content": f"Generate a detailed summary based on this dataset analysis: {summary}"}
            ]
        }

        response = httpx.post(api_url, json=payload, headers=headers)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

    except httpx.RequestError as e:
        print(f"Request error: {e}")
        sys.exit(1)
    except httpx.HTTPStatusError as e:
        print(f"HTTP status error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error generating narrative: {e}")
        sys.exit(1)

# Function to save README.md
def save_readme(narrative, output_dir):
    try:
        os.makedirs(output_dir, exist_ok=True)
        readme_path = os.path.join(output_dir, "README.md")
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write(narrative)
        print(f"README.md saved to {readme_path}")
    except Exception as e:
        print(f"Error saving README.md: {e}")

# Main function
def main():
    if len(sys.argv) != 2:
        print("Usage: python autolysis.py <dataset.csv>")
        sys.exit(1)

    file_path = sys.argv[1]
    if not Path(file_path).is_file():
        print(f"File not found: {file_path}")
        sys.exit(1)

    # Load the dataset
    data = load_csv(file_path)

    # Summarize the dataset
    summary = summarize_dataset(data)

    # Generate visualizations
    output_dir = "output"
    generate_visualizations(data, output_dir)

    # Generate narrative
    narrative = generate_narrative(summary)

    # Save README.md
    save_readme(narrative, output_dir)

if __name__ == "__main__":
    main()

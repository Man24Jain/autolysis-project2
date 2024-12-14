import os
import sys
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import httpx
import chardet

# Constants
API_URL = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
AIPROXY_TOKEN = "eyJhbGciOiJIUzI1NiJ9.eyJlbWFpbCI6IjIyZjMwMDA4NTJAZHMuc3R1ZHkuaWl0bS5hYy5pbiJ9.j20JpSkBVdolU8Npow1aEV4FY9e2NLC_Yvuzydx4viQ"

def load_data(file_path):
    """Load CSV data with encoding detection."""
    if not os.path.isfile(file_path):
        print(f"Error: File '{file_path}' not found.")
        sys.exit(1)
    with open(file_path, 'rb') as f:
        result = chardet.detect(f.read())
    encoding = result['encoding']
    print(f"Detected file encoding: {encoding}")
    return pd.read_csv(file_path, encoding=encoding)

def analyze_data(df):
    """Perform basic data analysis."""
    if df.empty:
        print("Error: Dataset is empty.")
        sys.exit(1)
    numeric_df = df.select_dtypes(include=['number'])  # Select only numeric columns
    analysis = {
        'summary': df.describe(include='all').to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'correlation': numeric_df.corr().to_dict()  # Compute correlation only on numeric columns
    }
    print("Data analysis complete.")
    return analysis

def visualize_data(df):
    """Generate and save visualizations."""
    sns.set(style="whitegrid")
    numeric_columns = df.select_dtypes(include=['number']).columns
    if numeric_columns.empty:
        print("No numeric columns found for visualization.")
        return
    for column in numeric_columns:
        plt.figure()
        sns.histplot(df[column].dropna(), kde=True)
        plt.title(f'Distribution of {column}')
        file_name = f'{column}_distribution.png'
        plt.savefig(file_name)
        print(f"Saved distribution plot: {file_name}")
        plt.close()

def generate_narrative(analysis):
    """Generate narrative using LLM."""
    headers = {
        'Authorization': f'Bearer {AIPROXY_TOKEN}',
        'Content-Type': 'application/json'
    }
    prompt = f"Provide a detailed analysis based on the following data summary: {analysis}"
    data = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": prompt}]
    }
    try:
        response = httpx.post(API_URL, headers=headers, json=data, timeout=30.0)
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']
    except httpx.HTTPStatusError as e:
        print(f"HTTP error occurred: {e}")
    except httpx.RequestError as e:
        print(f"Request error occurred: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    return "Narrative generation failed due to an error."

def main(file_path):
    print("Starting autolysis process...")
    df = load_data(file_path)
    print("Dataset loaded successfully.")
    
    print("Analyzing data...")
    analysis = analyze_data(df)
    
    print("Generating visualizations...")
    visualize_data(df)
    
    print("Generating narrative...")
    narrative = generate_narrative(analysis)
    
    if narrative != "Narrative generation failed due to an error.":
        with open('README.md', 'w') as f:
            f.write(narrative)
        print("Narrative successfully written to README.md.")
    else:
        print("Narrative generation failed. Skipping README creation.")
    
    print("Autolysis process completed.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python autolysis.py <file_path>")
        sys.exit(1)
    main(sys.argv[1])

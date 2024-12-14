import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import openai
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_fixed

# Load environment variables
load_dotenv()
API_PROXY_TOKEN = os.getenv("AIPROXY_TOKEN")

if not API_PROXY_TOKEN:
    raise EnvironmentError("AIPROXY_TOKEN not found in environment variables.")

openai.api_key = API_PROXY_TOKEN

# Ensure all required dependencies are installed
def ensure_dependencies():
    try:
        import seaborn
    except ImportError:
        os.system('pip install seaborn')
        print("Seaborn installed.")

ensure_dependencies()

@retry(stop=stop_after_attempt(3), wait=wait_fixed(5))
def analyze_with_llm(prompt):
    """Send a prompt to the AI proxy and return its response."""
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": "You are a data analysis assistant."},
                  {"role": "user", "content": prompt}],
        max_tokens=1500,
        temperature=0.7,
        timeout=120  # Extend timeout to allow detailed analysis
    )
    return response["choices"][0]["message"]["content"]

def generate_analysis(data):
    """Perform basic analysis and use LLM for insights."""
    # Basic pandas profiling
    description = data.describe(include='all').to_string()
    data_info = []
    data.info(buf=data_info.append)
    data_info = "\n".join(data_info)

    # Prompt construction
    llm_prompt = f"""
    I have the following dataset description:\n\n{description}\n\nAnd the following data info:\n\n{data_info}\n\n
    Please provide detailed insights, anomalies, patterns, and any recommended visualizations or analysis steps.
    Make the response engaging and unique.
    The response should include storytelling elements where relevant.
    """

    llm_response = analyze_with_llm(llm_prompt)
    return llm_response

def generate_visualizations(data, output_dir):
    """Generate and save visualizations based on the dataset."""
    sns.set(style="whitegrid")

    for column in data.select_dtypes(include=['number']).columns:
        plt.figure(figsize=(8, 6))
        sns.histplot(data[column], kde=True, color='blue')
        plt.title(f"Distribution of {column}")
        plt.xlabel(column)
        plt.ylabel("Frequency")
        output_path = os.path.join(output_dir, f"{column}_distribution.png")
        plt.savefig(output_path)
        plt.close()

    print(f"Visualizations saved to {output_dir}")

def main():
    if len(sys.argv) != 2:
        print("Usage: uv run autolysis.py <csv_filename>")
        sys.exit(1)

    csv_filename = sys.argv[1]

    if not os.path.exists(csv_filename):
        print(f"File not found: {csv_filename}")
        sys.exit(1)

    # Load the data
    try:
        data = pd.read_csv(csv_filename)
    except Exception as e:
        print(f"Error loading CSV: {e}")
        sys.exit(1)

    # Create output directory for visualizations
    output_dir = "visualizations"
    os.makedirs(output_dir, exist_ok=True)

    # Generate analysis
    try:
        llm_response = generate_analysis(data)
        print("\nLLM Analysis:\n", llm_response)

        # Write the LLM response to README.md
        with open("README.md", "w") as readme_file:
            readme_file.write("# Analysis Report\n\n")
            readme_file.write(llm_response)

        # Generate visualizations
        generate_visualizations(data, output_dir)

    except Exception as e:
        print(f"Error during analysis: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

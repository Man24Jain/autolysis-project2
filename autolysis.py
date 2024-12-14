import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import openai
import os

# Set OpenAI API key (make sure to set your key here)
openai.api_key = "eyJhbGciOiJIUzI1NiJ9.eyJlbWFpbCI6IjIyZjMwMDA4NTJAZHMuc3R1ZHkuaWl0bS5hYy5pbiJ9.j20JpSkBVdolU8Npow1aEV4FY9e2NLC_Yvuzydx4viQ"

def load_dataset(file_path):
    """Loads the dataset from the given file path."""
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        raise FileNotFoundError(f"Error loading file: {e}")

def generate_summary_statistics(df):
    """Generates summary statistics for the dataset."""
    summary = df.describe(include='all').T
    return summary

def detect_outliers(df):
    """Detects outliers in the dataset."""
    outlier_counts = {}
    for col in df.select_dtypes(include=['float', 'int']):
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        outliers = df[(df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)]
        outlier_counts[col] = len(outliers)
    return outlier_counts

def visualize_data(df):
    """Creates visualizations and saves them as PNG files."""
    os.makedirs("visualizations", exist_ok=True)

    # Correlation heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Correlation Heatmap")
    plt.savefig("visualizations/correlation_heatmap.png")

    # Pairplot
    sns.pairplot(df.select_dtypes(include=['float', 'int']).dropna(), diag_kind='kde')
    plt.savefig("visualizations/pairplot.png")

    # Outlier detection visualizations
    for col in df.select_dtypes(include=['float', 'int']):
        plt.figure(figsize=(8, 4))
        sns.boxplot(x=df[col])
        plt.title(f"Boxplot of {col}")
        plt.savefig(f"visualizations/boxplot_{col}.png")

    print("Visualizations saved in the 'visualizations' directory.")

def interact_with_llm(prompt):
    """Interacts with the LLM to generate narrative or code."""
    response = openai.ChatCompletion.create(
        model="gpt-4",  # Adjust as needed
        messages=[
            {"role": "system", "content": "You are an expert data analyst."},
            {"role": "user", "content": prompt}
        ]
    )
    return response['choices'][0]['message']['content']

def generate_readme(df, summary, outliers):
    """Generates a README.md file narrating the analysis."""
    narrative_prompt = f"""
    Analyze the following dataset summary and outlier counts:

    Summary:
    {summary}

    Outliers:
    {outliers}

    Generate a narrative about the key trends, correlations, and anomalies. Make it engaging and informative.
    """
    narrative = interact_with_llm(narrative_prompt)

    readme_content = f"""
    # Dataset Analysis

    ## Summary Statistics
    ```
    {summary.to_string()}
    ```

    ## Outlier Analysis
    ```
    {outliers}
    ```

    ## Narrative
    {narrative}

    ## Visualizations
    ![Correlation Heatmap](visualizations/correlation_heatmap.png)
    ![Pairplot](visualizations/pairplot.png)
    For individual boxplots, check the 'visualizations' folder.
    """
    with open("README.md", "w") as f:
        f.write(readme_content)

    print("README.md generated successfully.")

if __name__ == "__main__":
    # Input dataset file
    file_path = input("Enter the dataset file path (e.g., data.csv): ")

    try:
        # Load dataset
        df = load_dataset(file_path)

        # Generate summary statistics
        summary = generate_summary_statistics(df)

        # Detect outliers
        outliers = detect_outliers(df)

        # Visualize data
        visualize_data(df)

        # Generate README.md
        generate_readme(df, summary, outliers)

    except Exception as e:
        print(f"An error occurred: {e}")

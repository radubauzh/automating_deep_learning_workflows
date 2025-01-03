import os
import pickle
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from openai import OpenAI


def flatten_list(nested_list):
    return [item for sublist in nested_list for item in sublist]


def generate_gpt_analysis_report(
    csv_path, model="gpt-4o-mini", output_folder="Results"
):
    """
    Reads the CSV file, sends it to GPT-4O for analysis, and saves a Markdown report.

    :param csv_path: Path to the CSV file containing experiment results
    :param model:    GPT-4O model to use for analysis
    :param output_folder: Folder name where the Markdown report will be stored
    :return: None
    """
    # Read the CSV data
    df = pd.read_csv(csv_path)
    csv_text = df.to_csv(index=False)

    # Construct the prompt
    prompt = f"""
    You have CSV data from multiple experiments. Please analyze it thoroughly and produce a concise, 
    insightful report covering the following points:

    1. **Best Parameters**: Identify which parameters produced the best outcomes overall, especially include l2_sum_lambda (Additive Experiment) and l2_mul_lambda (Multiplicative), if both are 0 it's the no regularization Experiment.
    2. **Experiment Type Analysis**: Determine which experiment type performed best and provide insights 
    into their relative performance.
    3. **Top Experiments**: List the top 3 experiments overall, and also the best experiment within each 
    experiment type.
    4. **Detailed Insights**: Highlight any notable trends or observations from the data.
    5. **Recommendations**: Based on the analysis, provide recommendations for future experiments.

    Present the analysis in well-structured Markdown suitable for decision-making.

    CSV Data:
    {csv_text}
    """.strip()

    client = OpenAI(
        api_key=os.environ.get(
            "OPENAI_API_KEY"
        ), # Make sure to set your OpenAI API key as an environment variable
    )

    # Call the GPT-4O API
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are an advanced data analyst at MIT."},
            {"role": "user", "content": prompt},
        ],
    )

    analysis_report = response.choices[0].message.content

    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Save the analysis report to a Markdown file
    output_path = os.path.join(output_folder, "analysis_report.md")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(analysis_report)

    print(f"Analysis report saved to: {output_path}")


def analyze_results(pickle_file):
    # Load the results dictionary
    with open(pickle_file, "rb") as f:
        results = pickle.load(f)

    # Remove keys not needed for analysis
    keys_to_remove = [
        "margins_per_sample",
        "epoch_times",
        "test_confusion_matrices",
        "train_confusion_matrices",
        "norms",
        "model_state_name",
        "hp",
        "device",
    ]
    for key in keys_to_remove:
        if key in results:
            del results[key]

    # Build output folder name
    experiment_name = (
        "Analysis_Result_" + os.path.splitext(os.path.basename(pickle_file))[0]
    )
    output_dir = os.path.join("results", experiment_name)
    os.makedirs(output_dir, exist_ok=True)

    # Create CSV
    df = pd.DataFrame(results)
    csv_path = os.path.join(output_dir, "summary.csv")
    df.to_csv(csv_path, index=False)

    # Generate comprehensive GPT-4O report in Markdown format
    generate_gpt_analysis_report(
        csv_path=csv_path,
        model="gpt-4o-mini",
        output_folder=output_dir,  # Use the same output directory as the CSV file
    )


def main():
    parser = argparse.ArgumentParser(description="Analyze experiment results.")
    parser.add_argument(
        "--file",
        type=str,
        required=True,
        help="Path to the pickle file with experiment results",
    )
    args = parser.parse_args()
    analyze_results(args.file)


if __name__ == "__main__":
    main()

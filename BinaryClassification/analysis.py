import os
import pickle
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from openai import OpenAI
import ast 

def flatten_list(nested_list):
    return [item for sublist in nested_list for item in sublist]

def get_last_n_values(data, n=10):
    """Helper function to get last n values from a list or nested list."""
    if isinstance(data, list):
        if isinstance(data[0], list):
            return [item[-n:] for item in data]
        return data[-n:]
    elif isinstance(data, pd.Series):
        return data.apply(lambda x: x[-n:] if isinstance(x, list) else x)
    return data


def generate_gpt_analysis_report(
    csv_path, 
    model="gpt-4o-mini", 
    output_folder="Results", 
    additional_context=""
):
    """
    Reads the CSV file, sends it to GPT for analysis, and saves a Markdown report.

    :param csv_path: Path to the CSV file containing experiment results
    :param model:    GPT model to use for analysis
    :param output_folder: Folder name where the Markdown report will be stored
    :param additional_context: Extra text to add to the GPT prompt (e.g. info about plot filenames)
    :return: None
    """
    # Read the CSV data
    df = pd.read_csv(csv_path)
    csv_text = df.to_csv(index=False)

    # Construct the prompt
    prompt = f"""
    You are the best data analyst in the galaxy.
    You have CSV data from multiple experiments. Please analyze it thoroughly and produce a concise, 
    insightful report covering the following points:

    1. **Overall Performance**: Provide an overview of the performance metrics across all experiments, detect if the models learned or not and if they overfit or underfit.
    2. **Best Parameters**: Identify which parameters produced the best outcomes overall, especially include l2_sum_lambda (Additive Experiment) and l2_mul_lambda (Multiplicative). If both are 0, it's the no-regularization experiment.
    3. **Experiment Type Analysis**: Determine which experiment type performed best and provide insights into their relative performance.
    4. **Top Experiments**: List the top 3 experiments overall, and also the best experiment within each experiment type.
    5. **Detailed Insights**: Highlight any notable trends or observations from the data.
    6. **Recommendations**: Based on the analysis, provide recommendations for future experiments.

    Important: For all these points, if you mention an experiment or best parameters, it should always include ALL parameters used in the Experiment like learning rate, l2_sum_lambda, l2_mul_lambda, weight normalization (wn) and any other relevant parameters.

    **Important Notes on Regularization**:
    - Valid combinations of `l2_sum_lambda` and `l2_mul_lambda` are:
    - `l2_sum_lambda > 0`, `l2_mul_lambda = 0` (Additive Experiment)
    - `l2_sum_lambda = 0`, `l2_mul_lambda > 0` (Multiplicative Experiment)
    - `l2_sum_lambda = 0`, `l2_mul_lambda = 0` (No Regularization)
    - The combination `l2_sum_lambda > 0` and `l2_mul_lambda > 0` is not valid.

    {additional_context}

    Present the analysis in well-structured Markdown suitable for decision-making.

    CSV Data:
    {csv_text}
    """.strip()


    client = OpenAI(
        api_key=os.environ.get(
            "OPENAI_API_KEY"
        ),  # Make sure to set your OpenAI API key as an environment variable
    )

    # Call the GPT API
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
        "learning_rates",
        "misclassified_indices",
        "margins",
        "test_f1_scores",
        "train_f1_scores",
    ]
    for key in keys_to_remove:
        if key in results:
            del results[key]

    for key in results:
        if isinstance(results[key], str) and "[" in results[key]:
            results[key] = ast.literal_eval(results[key])

    # Truncate lists to last 10 values
    columns_to_truncate = [
        "train_epoch_losses",
        "train_epoch_l2_sum_losses",
        "train_epoch_l2_mul_losses",
        "train_batch_losses",
        "train_batch_l2_sum_losses",
        "train_batch_l2_mul_losses",
        "test_losses",
        "test_accuracies",
        "train_losses",
        "train_accuracies",
        "rho_values",
    ]
    for column in columns_to_truncate:
        if column in results:
            results[column] = get_last_n_values(results[column])

    # Build output folder name
    experiment_name = "Analysis_Result_" + os.path.splitext(os.path.basename(pickle_file))[0]
    output_dir = os.path.join("results", experiment_name)
    os.makedirs(output_dir, exist_ok=True)

    # Create CSV
    df = pd.DataFrame(results)
    csv_path = os.path.join(output_dir, "summary.csv")
    df.to_csv(csv_path, index=False)

    # Create a figure for Test Accuracy across experiments
    plt.figure(figsize=(10, 6))
    for i, row in df.iterrows():
        label_str = (
            f"Exp {i} | lr={row.get('lr','?')} | "
            f"sum={row.get('l2_sum_lambda','?')} | "
            f"mul={row.get('l2_mul_lambda','?')} | "
            f"wn={row.get('wn','?')}"
        )

        test_acc = row.get("test_accuracies", [])
        if not isinstance(test_acc, list):
            continue
        plt.plot(range(len(test_acc)), test_acc, marker='o', label=label_str)

    plt.title("Test Accuracy per Experiment (Last 10 Epochs)")
    plt.xlabel("Epoch (relative to the last 10)")
    plt.ylabel("Accuracy (%)")
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=1, fontsize='small')
    plt.tight_layout()
    accuracy_plot_path = os.path.join(output_dir, "accuracy_plot.png")
    plt.savefig(accuracy_plot_path, dpi=150)
    plt.close()

    # Create a figure for Test Loss across experiments
    plt.figure(figsize=(10, 6))
    for i, row in df.iterrows():
        label_str = (
            f"Exp {i} | lr={row.get('lr','?')} | "
            f"sum={row.get('l2_sum_lambda','?')} | "
            f"mul={row.get('l2_mul_lambda','?')} | "
            f"wn={row.get('wn','?')}"
        )

        test_loss = row.get("test_losses", [])
        if not isinstance(test_loss, list):
            continue
        plt.plot(range(len(test_loss)), test_loss, marker='o', label=label_str)

    plt.title("Test Loss per Experiment (Last 10 Epochs)")
    plt.xlabel("Epoch (relative to the last 10)")
    plt.ylabel("Loss")
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=1, fontsize='small')
    plt.tight_layout()
    loss_plot_path = os.path.join(output_dir, "loss_plot.png")
    plt.savefig(loss_plot_path, dpi=150)
    plt.close()

    # Generate GPT analysis report with references to these plot filenames
    plot_filenames = ["accuracy_plot.png", "loss_plot.png"]
    additional_context = (
        "We have also generated the following plots in the same folder: "
        f"{plot_filenames[0]}, {plot_filenames[1]}. "
        "Please embed them in your analysis report, for example:\n\n"
        f"![Accuracy Plot]({plot_filenames[0]})\n"
        f"![Loss Plot]({plot_filenames[1]})\n\n"
        "Make sure the legend is readable (it's placed at the bottom)."
    )

    # Generate comprehensive GPT-based report in Markdown format
    generate_gpt_analysis_report(
        csv_path=csv_path,
        model="gpt-4o-mini",
        output_folder=output_dir, 
        additional_context=additional_context
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

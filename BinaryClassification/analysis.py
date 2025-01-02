import os
import pickle
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def flatten_list(nested_list):
    return [item for sublist in nested_list for item in sublist]

def analyze_results(pickle_file):
    # Load the results dictionary
    with open(pickle_file, 'rb') as f:
        results = pickle.load(f)
        keys_to_remove = ['margins_per_sample', 'epoch_times', 'test_confusion_matrices', 'train_confusion_matrices', 'norms', 'model_state_name', 'hp', 'device']
    
    for key in keys_to_remove:
        if key in results:
            del results[key]   

    # Extract the filename without extension to use as the folder name
    experiment_name = "Results_" + os.path.splitext(os.path.basename(pickle_file))[0]
    output_dir = os.path.join('results', experiment_name)
    os.makedirs(output_dir, exist_ok=True)

    # Convert all data to a CSV file
    df = pd.DataFrame(results)
    csv_path = os.path.join(output_dir, 'summary.csv')
    df.to_csv(csv_path, index=False)

    # Standard plots
    train_acc = flatten_list(results.get('train_accuracies', []))
    test_acc = flatten_list(results.get('test_accuracies', []))
    epochs = range(1, len(train_acc) + 1)

    plt.figure()
    plt.plot(epochs, train_acc, label='Train Accuracy')
    plt.plot(epochs, test_acc, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title(f'Train vs. Test Accuracy: {experiment_name}')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'accuracy_plot.png'))
    plt.close()

    # Write a simple text summary
    with open(os.path.join(output_dir, 'analysis_summary.txt'), 'w') as f:
        f.write(f"Experiment: {experiment_name}\n")
        if len(test_acc) > 0:
            f.write(f"Best Test Accuracy: {max(test_acc):.2f}%\n")
        else:
            f.write("No test accuracies found.\n")

def main():
    parser = argparse.ArgumentParser(description="Analyze experiment results.")
    parser.add_argument('--file', type=str, required=True, help="Path to the pickle file with experiment results")
    args = parser.parse_args()
    analyze_results(args.file)

if __name__ == "__main__":
    main()

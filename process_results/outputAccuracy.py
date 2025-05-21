import os
import csv
import pandas as pd
from pathlib import Path
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--filepath", required=True, type=Path, help="Path to the model results CSV file")
    args = parser.parse_args()
    main_with_args(args.data)

def main_with_args(file_path: Path):
    model_name = file_path.stem  # Gets filename without extension

    # Read the CSV file
    df = pd.read_csv(file_path)

    # Calculate the accuracy percentage for each 'Visual' type
    accuracy_data = df[df['correctness'] == True].groupby('visual').size()
    total_data = df.groupby('visual').size()
    accuracy_percentage = (accuracy_data / total_data) * 100
    accuracy_percentage = accuracy_percentage.sort_index()

    # Check if the output file exists
    output_file = Path("accuracy_results.csv")
    file_exists = output_file.exists()

    # Write the results to a CSV file
    with output_file.open(mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(['Model_name', 'Image Style', 'Accuracy'])
        for index, value in accuracy_percentage.items():
            writer.writerow([model_name, index, f"{value:.2f}%"])

if __name__ == "__main__":
    main()

import os
import pandas as pd
import csv
import os

file_path = ''
model_name = os.path.splitext(os.path.basename(file_path))[0]

# Read the CSV file
df = pd.read_csv(file_path)

# Calculate the accuracy percentage for each 'Visual' type
accuracy_data = df[df['correctness'] == True].groupby('visual').size()
total_data = df.groupby('visual').size()
accuracy_percentage = (accuracy_data / total_data) * 100
accuracy_percentage = accuracy_percentage.sort_index()

# Check if the output file exists
file_exists = os.path.isfile('accuracy_results.csv')

# Write the results to a CSV file
with open('accuracy_results.csv', mode='a', newline='') as file:
    writer = csv.writer(file)
    if not file_exists:
        writer.writerow(['Model_name', 'Image Style', 'Accuracy'])
    for index, value in accuracy_percentage.items():
        writer.writerow([model_name,index, f"{value:.2f}%"])

import gzip
import json
from pathlib import Path
from typing import Callable
import os
import argparse
from datasets import load_dataset

def f_image(input_string: str,model_name: str,) -> str:
    model_name = model_name.lower()
    if model_name == 'idefics3' or model_name == 'idefics2':
        print(input_string)
        input_string = input_string.rsplit('Assistant:',1)[1]
        if 'Answer:' in input_string:
            input_string = input_string.split('Answer:')[1]
    elif model_name == 'llavanext':
        input_string = input_string.rsplit('[/INST]',1)[1]
        input_string = input_string.split('<\\s>')[0]
    input_string = input_string.strip().lstrip()
    input_string = input_string.strip('.')
    return input_string

def check_answers_image(p: Path, f: Callable[[str], str],model_name: str) -> dict:
    all_results = {}
    for file in p.glob("*.json.gz"):
        with gzip.open(file, "rt", encoding="utf-8") as infile:
            data = json.load(infile)
            file_name = data.get("extras", {}).get("file_name", None)
            answer = data.get("extras", {}).get("answer", None)
            completion = data.get("completions", [])[0]
            modified_text = f(completion["text"],model_name)
            all_results[file_name] = (modified_text, answer)
    return all_results

def run_image(result_dir, model_name):
    model_output = check_answers_image(Path(result_dir), f_image, model_name)
    print("model_output", model_output)
    for file_name, (output, answer) in model_output.items():
        rule_id, visual = file_name.split('_')
        output = output.split('.')[0]
        # print('answer:', answer)
        # print('output:', output)
        # print(f'data {file_name}:{answer == output}')
        
        # Write to CSV
        csv_file = f'{result_dir}.csv'
        if not os.path.exists(csv_file):
            with open(csv_file, 'w') as f:
                f.write('rule_id,visual,answer,most_frequent_output,correctness\n')
        with open(csv_file, 'a') as f:
            f.write(f'{rule_id},{visual},{answer},{output},{answer == output}\n')

def main():
    parser = argparse.ArgumentParser(description="Run image analysis and generate CSV results.")
    parser.add_argument('--result_dir', type=str, required=True, help='Directory containing the model output files')
    parser.add_argument('--model_name', type=str, required=True, help='The model name or identifier')
    args = parser.parse_args()
    
    run_image(args.result_dir, args.model_name)

if __name__ == "__main__":
    main()

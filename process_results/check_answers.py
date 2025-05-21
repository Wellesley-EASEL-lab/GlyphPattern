import gzip
import json
import os
import csv
import argparse
from pathlib import Path
from typing import Callable, Optional


def f_image(input_string: str, model_name: str) -> Optional[str]:
    model_name = model_name.lower()
    if model_name in ['idefics3', 'idefics2']:
        input_string = input_string.rsplit('Assistant:', 1)[-1]
        if 'Answer:' in input_string:
            input_string = input_string.split('Answer:')[1]
    elif model_name == 'llavanext':
        input_string = input_string.rsplit('[/INST]', 1)[-1]
        input_string = input_string.split('<\\s>')[0]
    elif model_name == 'gpt-4o':
        if 'Statement' in input_string:
            input_string = input_string.split()[1]
    input_string = input_string.strip().lstrip().strip('.')
    return input_string


def f_image_cot(input_string: str, model_name: str) -> Optional[str]:
    if 'Answer:' in input_string:
        input_string = input_string.split('Answer:')[1]
        input_string = input_string.split(".")[0].replace('*', '').strip().lstrip()
        if input_string not in ['A', 'B', 'C', 'D']:
            return None
        return input_string
    return None


def check_answers_image(p: Path, f: Callable[[str, str], Optional[str]], model_name: str, is_cot: bool) -> dict:
    all_results = {}
    for file in p.glob("*.json.gz"):
        with gzip.open(file, "rt", encoding="utf-8") as infile:
            data = json.load(infile)
            file_name = data.get("extras", {}).get("file_name", None)
            answer = data.get("extras", {}).get("answer", None)
            completion = data.get("completions", [])[0]
            model_output = completion["text"]
            processed_output = f(model_output, model_name)
            if is_cot:
                all_results[file_name] = (model_output, processed_output, answer)
            else:
                all_results[file_name] = (processed_output, answer)
    return all_results


def run_image(result_dir: str, model_name: str):
    model_output = check_answers_image(Path(result_dir), f_image, model_name, is_cot=False)
    csv_file = f'{model_name}_zeroshot.csv'
    file_exists = os.path.exists(csv_file)

    with open(csv_file, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['rule_id', 'visual', 'answer', 'most_frequent_output', 'correctness'])
        for file_name, (output, answer) in model_output.items():
            rule_id, visual = file_name.split('_')
            writer.writerow([rule_id, visual, answer, output, answer == output])


def run_image_cot(result_dir: str, model_name: str):
    model_output = check_answers_image(Path(result_dir), f_image_cot, model_name, is_cot=True)
    csv_file = f'{model_name}_zeroshot_cot.csv'
    file_exists = os.path.exists(csv_file)

    with open(csv_file, 'a', newline='') as f:
        writer = csv.writer(f, quotechar='"', quoting=csv.QUOTE_ALL)
        if not file_exists:
            writer.writerow(['rule_id', 'visual', 'answer', 'cot_output', 'choice_output', 'correctness'])
        for file_name, (cot_output, choice_output, answer) in model_output.items():
            rule_id, visual = file_name.split('_')
            writer.writerow([rule_id, visual, answer, cot_output, choice_output, answer == choice_output])


def main():
    parser = argparse.ArgumentParser(description="Run image analysis and generate CSV results.")
    parser.add_argument('--result_dir', type=str, required=True, help='Directory containing model output files')
    parser.add_argument('--model_name', type=str, required=True, help='The model name or identifier')
    parser.add_argument('--cot', action='store_true', help='Use CoT extraction mode')
    args = parser.parse_args()

    if args.cot:
        run_image_cot(args.result_dir, args.model_name)
    else:
        run_image(args.result_dir, args.model_name)


if __name__ == "__main__":
    main()

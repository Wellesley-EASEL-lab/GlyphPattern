import os
import random
import string
import pandas as pd
import json
from PIL import Image
import os
import datasets

image_dir='dataset_images'
rules_df = pd.read_csv('final_335_rules.csv')
with open('no_dup.json', 'r') as f:
   no_dup = json.load(f)

variable_name = 'Characters'
variable_name_intext = 'characters'

def generate_groundtruth_descriptions():
    path_text={}

    for image_name in sorted(os.listdir(image_dir), key=lambda x: int(x.split('_')[0])):
        if not image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
        image_path = os.path.join(image_dir, image_name)
        rule_id, visual = os.path.splitext(image_name)[0].split('_')   
        description = rules_df.loc[rules_df['ID'] == int(rule_id), 'description'].values[0].replace('VARIABLE', variable_name_intext)
        if visual=="color":
            text = variable_name + ' colored red in the image are ' + variable_name_intext + ' that '+description.strip().rstrip('.')+'.'
        elif visual == "leftright":
            text = variable_name + ' on the right side in the image are ' + variable_name_intext + ' that '+description.strip().rstrip('.')+'.'
        elif visual=="circle":
            text = variable_name + ' inside the circle in the image are ' + variable_name_intext + ' that '+description.strip().rstrip('.')+'.'
        else:
            raise ValueError("Invalid visual type")
        path_text[image_path] = {'prompt': text, 'answer':description}
    return dict(sorted(path_text.items()))

def generate_multiple_choice_descriptions(num_of_choice):
    path_text = {}
    
    for image_name in sorted(os.listdir(image_dir), key=lambda x: int(x.split('_')[0])):
        if not image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
        image_path = os.path.join(image_dir, image_name)

        rule_id, visual = os.path.splitext(image_name)[0].split('_')   
        description = rules_df.loc[rules_df['ID'] == int(rule_id), 'description'].values[0]
        other_descriptions = []

        question_text = "Which characters are "
        for other_rule_id in no_dup[rule_id]:
            other_rule_id = int(other_rule_id)
            other_description = rules_df.loc[rules_df['ID'] == other_rule_id, 'description'].values[0]
            other_descriptions.append(other_description)

        if visual == "color":
            question_text += "colored red in the image?\n"
            correct_choice = variable_name + ' that ' + description.replace('VARIABLE', variable_name_intext) +"."
        elif visual == "leftright":
            question_text += "on the right side in the image?\n"
            correct_choice = variable_name + ' that ' + description.replace('VARIABLE', variable_name_intext) +"."
        elif visual == "circle":
            question_text += "inside the circle in the image?\n"
            correct_choice = variable_name + ' that ' + description.replace('VARIABLE', variable_name_intext)  +"."
        else:
            raise ValueError("Invalid visual type")
        
        other_choices = random.sample(other_descriptions, num_of_choice - 1)
        other_choices = [variable_name + ' that ' + choice.replace('VARIABLE', variable_name_intext) + "." for choice in other_choices]
        
        # Combine correct choice and other choices, shuffle before adding labels
        all_choices = [correct_choice] + other_choices
        random.shuffle(all_choices)
        labeled_choices = [f"{string.ascii_uppercase[i]}. {choice}" for i, choice in enumerate(all_choices)]
        correct_answer = string.ascii_uppercase[all_choices.index(correct_choice)]
        
        options = "\n".join(labeled_choices)
        question_text += options
        
        path_text[image_path] = {'question': question_text, 'answer': correct_answer}
    
    return dict(sorted(path_text.items()))


def create_prompts_multiplechoice(path_multiple_text_dict):
    prompts = []
    for image_path,qa in path_multiple_text_dict.items():
        question=qa['question']
        answer = qa['answer']
        file_name = os.path.basename(image_path)
        file_name =  os.path.splitext(file_name)[0]
        image = Image.open(image_path).convert("RGB") 
        prompts.append({ "file_name":file_name, "prompt": question, "answer": answer, "images": image})
    return prompts

def create_prompts_groundtruth(path_multiple_text_dict):
    prompts = []
    for image_path,qa in path_multiple_text_dict.items():
        description=qa['prompt']
        answer = qa['answer']
        file_name = os.path.basename(image_path)
        file_name =  os.path.splitext(file_name)[0]
        image = Image.open(image_path).convert("RGB") 
        prompts.append({ "file_name":file_name, "prompt": description, "answer": answer,"images": image})
    return prompts
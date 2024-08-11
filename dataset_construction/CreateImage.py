#!/usr/bin/env python3
#have to do this on my local computer though, since the fontbook exist on my computer only.
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties, findfont

characters_df = pd.read_csv('dataset_construction/all-script-fontname-data.csv')
rules_df = pd.read_csv('dataset_construction/final_rules.csv')


# Function to map binary string to characters and create two groups
def map_binary_to_characters(dataframe_characters, dataframe_rules):
    mapped_characters = {}

    # Iterate over the rows of the rules dataframe
    for index, rule_row in dataframe_rules.iterrows():
        rule_id = rule_row['ID']
        description = rule_row['description']
        script_name = rule_row['script']
        characters_row = dataframe_characters[dataframe_characters['Name'] == script_name].iloc[0]

        characters = characters_row['Characters'].split(',')
        binary_string = rule_row['RuleBinary']

        if rule_id not in mapped_characters:
            mapped_characters[rule_id] = {'Description': description, 'Script Name':script_name,'Selected': [], 'NotSelected': []}

        for char, bit in zip(characters, binary_string):
            if bit == '1':
                mapped_characters[rule_id]['Selected'].append(char)
            else:
                mapped_characters[rule_id]['NotSelected'].append(char)

    mapped_characters_sorted = dict(sorted(mapped_characters.items()))
    return mapped_characters_sorted


# Function to create image in color format
def create_image_color(original_characters, selected_characters, font_name):
    font_path = findfont(FontProperties(family=font_name))
    
    rows = [original_characters[i:i + 10] for i in range(0, len(original_characters), 10)]

    fig, ax = plt.subplots(figsize=(15,12))
    ax.set_aspect('equal')

    for i, row in enumerate(rows):
        for j, char in enumerate(row):
            color = 'red' if char in selected_characters else 'black'
            ax.text(j + 0.5, -i - 0.5, char, ha='center', va='center', fontsize=30, color=color, fontproperties={'fname': font_path})

    ax.set_xticks(range(10))
    ax.set_yticks(range(-len(rows), 0))
    ax.grid(color='black', linestyle='-', linewidth=1, alpha=0.5)
    ax.axis('off')

    return fig



# Function to create image with included characters on the left and excluded on the right
def create_image_leftright(included_characters_decoded, excluded_characters_decoded, fontname):
    characters_per_row = 5
    rows_included = [included_characters_decoded[i:i + characters_per_row] for i in range(0, len(included_characters_decoded), characters_per_row)]
    rows_excluded = [excluded_characters_decoded[i:i + characters_per_row] for i in range(0, len(excluded_characters_decoded), characters_per_row)]

    total_rows = max(len(rows_included), len(rows_excluded))
    fig, ax = plt.subplots(figsize=(18, total_rows * 0.5))
    ax.set_aspect('equal')

    for i in range(total_rows):
        included_row = rows_included[i] if i < len(rows_included) else []
        excluded_row = rows_excluded[i] if i < len(rows_excluded) else []

        for j, char in enumerate(excluded_row):
            ax.text(j - characters_per_row , -i - 0.5, char, ha='center', va='center', fontsize=10, color='black', fontname=fontname)
    
        for j, char in enumerate(included_row):
            ax.text(j+1, -i - 0.5, char, ha='center', va='center', fontsize=10, color='black', fontname=fontname)

    ax.set_xticks(range(-characters_per_row, characters_per_row))
    ax.set_yticks(range(-total_rows, 0))
    
    ax.axvline(0, color='black', linestyle='--', linewidth=1)  # Vertical line in the middle
    ax.grid(color='black', linestyle='-', linewidth=1, alpha=0.5)
    ax.axis('off')

    return fig

def create_image_circle(inside_characters, outside_characters, fontname):
    fontsize = 50 - len(outside_characters)* 0.1
    circle_radius = 0.7
    #num_segments = max(len(inside_characters),len(outside_characters))

    fig, ax = plt.subplots(figsize=(25, 25))
    ax.set_aspect('equal')
    ax.grid(color='black', linestyle='-', linewidth=1, alpha=0.5)

    # Draw the circle outline
    circle = plt.Circle((0, 0), circle_radius, linewidth=2, edgecolor='black', facecolor='none')
    ax.add_patch(circle)

    # Draw characters inside the circle
    for i, char in enumerate(inside_characters):
        theta = (i % len(inside_characters)) * (2 * np.pi / len(inside_characters))
        r = circle_radius - 0.15 
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        ax.text(x, y, char, ha='center', va='center', fontsize=fontsize, color='black', fontname=fontname)

    # Draw characters surrounding the circle
    far = 0.2
    for i, char in enumerate(outside_characters):
        theta = (i % len(outside_characters)) * (2 * np.pi / len(outside_characters))
        r = circle_radius + far
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        ax.text(x, y, char, ha='center', va='center', fontsize=fontsize, color='black', fontname=fontname)

    ax.set_xlim(-circle_radius - far, circle_radius + far)
    ax.set_ylim(-circle_radius - far, circle_radius + far)
    ax.axis('off')

    return fig

def run_all():
    # Iterate over the rules, create and save images
    for i, (rule_id, data) in enumerate(map_binary_to_characters(characters_df, rules_df).items()):
        if i>1:break
        script_name = data['Script Name']
        fontname = characters_df.loc[characters_df['Name'] == script_name, 'Fontname'].iloc[0]
        # Extract characters in their original order from the script
        original_characters = characters_df[characters_df['Name'] == script_name]['Characters'].iloc[0].split(',')
        
        # Decode Unicode characters
        original_characters_decoded = [chr(int(char.replace('U+', ''), 16)) for char in original_characters]
        selected_characters_decoded = [chr(int(char.replace('U+', ''), 16)) for char in data['Selected']]
        excluded_characters_decoded = [chr(int(char.replace('U+', ''), 16)) for char in data['NotSelected']]

        # Create image
        image1 = create_image_color(original_characters_decoded, selected_characters_decoded, fontname)
        image2 = create_image_leftright(selected_characters_decoded, excluded_characters_decoded,fontname)
        image3 = create_image_circle(selected_characters_decoded, excluded_characters_decoded, fontname)

        # Save image
        image1.savefig(f'datast_construction/images/{rule_id}_color.png', dpi = 300, bbox_inches='tight')
        plt.close(image1)
        image2.savefig(f'datast_construction/images/{rule_id}_leftright.png', dpi = 300, bbox_inches='tight')
        plt.close(image2)
        image3.savefig(f'datast_construction/images/{rule_id}_circle.png',  dpi = 300,bbox_inches='tight')
        plt.close(image3)

run_all()
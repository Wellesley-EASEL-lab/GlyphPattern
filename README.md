## Dataset Construction

1. The images are under dataset_construction/dataset_images
The code for generating the images is in dataset_construction/CreateImage.py 
Note: You might need to install a variety of fonts in the system in order to run the code.

2. Run generate_qa.ipynb to generate the full dataset and save them to a local dataset.


## Running Experiments
The codes for running each model on GlyphPattern are in /run_model directory. For each model, run the code file named by the model name.
argument --fewshot-prompt: choose from '1shot', '3shot', '5shot', None(leave empty for zero-shot)

### Multiple Choice
eg. Run GPT-4o using gpt4o_vision.py few-shot with three examples.
    
    ```bash
    python3 -m prl_ml.batched_lm_generation.gpt4o_vision \
        --dataset "disk:path_to_GlyphPattern_repo/dataset_construction/GlyphPattern/four_choice" \
        --output-dir gpt4o-4choice-3shot \
        --model-name gpt-4o \
        --temperature 0 \
        --batch-size 1 \
        --completion-limit 1 \
        --max-tokens 10 \
        --prompt-keys prompt,images \
        --fewshot-prompt 3shot \
        --extra-columns file_name,answer
    ```
    
For Idefics models and other chat models that support AutoModelForVision2Seq Class, run the idefics_vision code file with output-dir and model-name replaced.
eg.Run Idefics2 from HuggingFaceM4/idefics2-8b few shot with one example:

    ```bash
    python3 -m run_model.idefics_vision \
        --dataset "disk:path_to_GlyphPattern_repo/dataset_construction/GlyphPattern/four_choice" \
        --output-dir Idefics2-4choice-zeroshot \
        --model-name HuggingFaceM4/idefics2-8b \
        --temperature 0 \
        --batch-size 1 \
        --completion-limit 1 \
        --prompt-keys prompt,images \
        --fewshot-prompt 1shot \
        --extra-columns file_name,answer
    ```

### Few-shot Free Response
We support few-shot free response for the best two performing models, gpt-4o and gemini 1.5.
Run code files with suffix freeresponse, and use dataset-split gt_descriptions.

For example:
    ```bash
    python3 -m run_model.gpt4o_freeresponse \
        --dataset "disk:path_to_GlyphPattern_repo/dataset_construction/GlyphPattern/gt_descriptions" \
        --output-dir gpt4o-free \
        --model-name gpt-4o \
        --temperature 0 \
        --batch-size 1 \
        --completion-limit 1 \
        --max-tokens 500 \
        --prompt-keys prompt,images \
        --extra-columns file_name,answer
    ```

### Few-shot Chain-Of-Thought
For example:
    ```
    python3 -m run_model.gpt4o_vision_cot \
        --dataset "disk:path_to_GlyphPattern_repo/dataset_construction/GlyphPattern/four_choice" \
        --output-dir gpt4o-4choice-fewshot-cot \
        --model-name gpt-4o-2024-05-13 \
        --temperature 0 \
        --batch-size 1 \
        --completion-limit 1 \
        --max-tokens 1024 \
        --prompt-keys prompt,images \
        --fewshot-prompt 3shot \
        --extra-columns file_name,answer
    ```

## Process Model Results:
The model results csv files are in model_results directory.

To process the generated results from running experiment above, run the check_answers.py in process_result directory. Replace the result_dir with the model output directory created by running batched_lm_generation.

For example, to check the outputs of running molmod zero shot multiple choice:
    ```bash
        python process_results/check_answers.py \
        --result_dir molmod-4choice-zeroshot \
        --model_name molmod \
    ```
To check the outputs of model results with chain-of-thought:
    ```bash
        python process_results/check_answers.py \
        --result_dir geminipro-4choice-zeroshot-cot \
        --model_name geminipro \
        --cot
    ```
This produces a csv file at where the result directory was.

To log accuracy by on visual output, run outputAccuracy.py with the correct file_path.
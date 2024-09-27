## Dataset Construction

1. The images are under dataset_construction/dataset_images
The code for generating the images is in dataset_construction/CreateImage.py 
Note: You might need to install a variety of fonts in the system in order to run the code.

2. Run generate_qa.ipynb to generate the full dataset and save them to a local dataset.


## Running Experiments
Our code for running experiments rely on prl_ml(https://github.com/nuprl/prl_ml), which supports resumption.

Keep batch-size as 1 for all experiments, as batching is not possible for those models.

To run the experiments, first clone the prl_ml repository. Then drag all the files in run_model directory into prl_ml/batched_lm_generation directory. cd into the repository to run the generations with the commands below.

### Multiple Choice
For each model, run the code file named by the model name.
argument --fewshot-prompt: choose from '1shot', '3shot', '5shot', None(leave empty for zero-shot)

1. GPT-4o
eg. Run GPT-4o using gpt4o_vision.py few-shot with three examples.
    
    ```bash
    python3 -m prl_ml.batched_lm_generation.gpt4o_vision \
        --dataset "disk:path_to_GlyphPattern_repo/dataset_construction/GlyphPattern/four_choice" \
        --output-dir gpt4o-4choice-zeroshot \
        --model-name gpt-4o \
        --temperature 0 \
        --batch-size 1 \
        --completion-limit 1 \
        --max-tokens 10 \
        --prompt-keys prompt,images \
        --extra-columns file_name,answer
    ```

2. Gemini 1.5
eg. Run Gemini1.5Pro using geminipro_vision.py few-shot with five examples.

    ```bash
    python3 -m batched_lm_generation.geminipro_vision \
        --dataset "disk:path_to_GlyphPattern_repo/dataset_construction/GlyphPattern/four_choice" \
        --output-dir geminipro-4choice-zeroshot \
        --model-name gemini-1.5-pro \
        --temperature 0 \
        --batch-size 1 \
        --completion-limit 1 \
        --max-tokens 10 \
        --prompt-keys prompt,images \
        --fewshot-prompt 5shot \
        --extra-columns file_name,answer
    ```

For Idefics models and other chat models that support AutoModelForVision2Seq Class, run the idefics_vision code file with output-dir and model-name replaced.
3. Idefics2
eg.Run Idefics2 from HuggingFaceM4/idefics2-8b few shot with one example:

    ```bash
    python3 -m batched_lm_generation.idefics_vision \
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

4. Idefics3
eg.Run Idefics3 from HuggingFaceM4/Idefics3-8B-Llama3 zero shot:

    ```bash
    python3 -m batched_lm_generation.idefics_vision \
        --dataset "disk:path_to_GlyphPattern_repo/dataset_construction/GlyphPattern/four_choice" \
        --output-dir Idefics2-4choice-zeroshot \
        --model-name HuggingFaceM4/idefics2-8b \
        --temperature 0 \
        --batch-size 1 \
        --completion-limit 1 \
        --prompt-keys prompt,images \
        --extra-columns file_name,answer
    ```

5. LlavaNext
eg. Run LlavaNext from llava-hf/llava-v1.6-mistral-7b-hf Zero-shot
    ```bash
    python3 -m batched_lm_generation.llavanext_vision \
        --dataset "disk:path_to_GlyphPattern_repo/dataset_construction/GlyphPattern/four_choice" \
        --output-dir llavanext-4choice-zeroshot \
        --model-name llava-hf/llava-v1.6-mistral-7b-hf \
        --temperature 0 \
        --batch-size 1 \
        --completion-limit 1 \
        --prompt-keys prompt,images \
        --extra-columns file_name,answer
    ```

6. InstructBlip
Run InstructBlip from Salesforce/instructblip-vicuna-7b, only zero-shot is supported.
    ```bash
    python3 -m batched_lm_generation.instructblip_vision \
        --dataset "disk:path_to_GlyphPattern_repo/dataset_construction/GlyphPattern/four_choice" \
        --output-dir instructblip-4choice-zeroshot \
        --model-name Salesforce/instructblip-vicuna-7b \
        --temperature 0 \
        --batch-size 1 \
        --completion-limit 1 \
        --prompt-keys prompt,images \
        --extra-columns file_name,answer
    ```

### Few-shot Free Response
We support few-shot free response for the best two performing models, gpt-4o and gemini 1.5.
Run code files with suffix freeresponse, and use dataset-split gt_descriptions.

1. GPT-4o
    ```bash
    python3 -m batched_lm_generation.gpt4o_freeresponse \
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

2. Gemini1.5Pro
    ```bash
    python3 -m batched_lm_generation.geminipro_freeresponse \
        --dataset "disk:path_to_GlyphPattern_repo/dataset_construction/GlyphPattern/gt_descriptions" \
        --output-dir geminipro-free-leftright \
        --model-name gemini-1.5-pro \
        --temperature 0 \
        --batch-size 1 \
        --completion-limit 1 \
        --max-tokens 500 \
        --prompt-keys prompt,images \
        --extra-columns file_name,answer
    ```

## Process Model Results:
The model results csv files are in model_results directory.

To process the generated results from running experiment above, run the check_answers.py in process_result directory. Replace the result_dir with the model output directory created by running batched_lm_generation.

For example, to check the outputs of running llavanext zero shot multiple choice:
    ```bash
        python process_results/check_answers.py \
        --result_dir llavanext-4choice-zeroshot \
        --model_name llavanext
    ```
This produces a csv file at where the result directory was.

To calculate accuracy base on visual output, run outputAccuracy.py with the file_path replaced.
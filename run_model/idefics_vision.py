from .base import GeneratorBase, partial_arg_parser, stop_at_stop_token
import torch
from PIL import Image
from io import BytesIO
from datasets import load_dataset, Dataset
from transformers import AutoProcessor, AutoModelForVision2Seq
from transformers.image_utils import load_image
from typing import List, Tuple
import torch
from collections import defaultdict
from .glyphpattern_utils import create_fewshot_dataset,glyph_pattern_argparser

def prompt_to_messages(prompt: str) -> List[dict]:
    return [
        {
            "role": "user", 
            "content": [ 
                { "type": "image" },
                {
                    "type": "text",
                    "text": f"{prompt}"
                }
            ]
        }
    ]

class VisionModel(GeneratorBase):
    model_name: str
    model_kwargs: dict
    processor: AutoProcessor
    model: AutoModelForVision2Seq

    def __init__(self, model_name: str, **super_args):
        super().__init__(**super_args)
        self.model_name = model_name

    def init_model(self):
        if self.model_name == "HuggingFaceM4/idefics2-8b":
            self.processor = AutoProcessor.from_pretrained(self.model_name,size= {"longest_edge": 700, "shortest_edge": 378})
        if self.model_name == "HuggingFaceM4/Idefics3-8B-Llama3":
            self.processor = AutoProcessor.from_pretrained(self.model_name,size= {"longest_edge": 3*364})
        else:
            self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.model = AutoModelForVision2Seq.from_pretrained(
            self.model_name,
            device_map="cuda"
        )

    # Each prompt is a tuple with a text prompt and an image.
    @torch.no_grad
    def batch_generate(
        self,
        prompts: List[Tuple[str, Image.Image]],
        top_p: float,
        temperature: float,
        max_tokens: int,
        stop: List[str],
    ) -> List[str]:
        question_type_dict = {
            "Which characters are on the right side in the image":'leftright',
            "Which characters are colored red in the image":'color',
            "Which characters are inside the circle in the image":'circle',}
        text = prompts[0][0]
        question_type = question_type_dict[text.split('?\n')[0]]
        few_shot_messages = []
        images_list=[]
        if fewshot_dataset:
            #filter out the ones that have questiontype in file_name
            fewshot_typed = [x for x in fewshot_dataset if question_type in x['file_name']]
            for example in fewshot_typed:
                example_text = example['prompt']
                example_image = example['images']
                images_list.append(example_image)
                example_answer = example['answer']
                message = prompt_to_messages(example_text)[0]
                few_shot_messages.append(message)
                assistant_message = {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": example_answer},
                    ]
                }
                few_shot_messages.append(assistant_message)

        new_message = prompt_to_messages(text)[0]
        few_shot_messages.append(new_message)
        images_list.append(prompts[0][1])
        prompts = self.processor.apply_chat_template(few_shot_messages, add_generation_prompt=True)
        inputs = self.processor(text=prompts, images=images_list, return_tensors="pt").to(self.model.device)
        generated_ids = self.model.generate(**inputs, max_new_tokens=max_tokens)
        generated_texts = self.processor.batch_decode(generated_ids, skip_special_tokens=True,do_sample=False)
        return generated_texts
    
def main(): 
    parser = partial_arg_parser()
    parser.add_argument("--model-name", type=str, required=True)
    glyph_pattern_argparser(parser)
    args = parser.parse_args()

    EXTRA_ARGS = ["model_name", "fewshot_prompt"]
    super_args = {k: v for (k, v) in vars(args).items() if k not in EXTRA_ARGS}

    generator = VisionModel(model_name=args.model_name, stop=[], **super_args)
    global fewshot_dataset
    fewshot_dataset = create_fewshot_dataset(dataset=generator.actual_dataset, prompt_kind = args.fewshot_prompt)

    generator.generate_all()

if __name__ == "__main__":
    main()

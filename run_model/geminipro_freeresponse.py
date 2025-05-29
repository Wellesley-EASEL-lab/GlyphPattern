from .base import GeneratorBase, partial_arg_parser
from PIL import Image
from typing import List, Tuple
import google.generativeai as genai
import os
from typing import List, Tuple
import datasets
from google.generativeai.types import content_types
from .glyphpattern_utils import create_fewshot_dataset,glyph_pattern_argparser

class GeminiModel(GeneratorBase):
    model_name: str
    model_kwargs: dict
    model: genai.GenerativeModel

    def __init__(self, model_name: str, **super_args):
        super().__init__(**super_args)
        self.model_name = model_name
        self.model = None

    def init_model(self):
        genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
        self.model = genai.GenerativeModel(model_name="gemini-1.5-pro-001",system_instruction='Come up with a descriptions for all of the characters on the right side in the image. Your description should be true for all of the characters on the right side and none of the characters on the left side.')
    
    def update_system_instruction(self, new_instruction):
        self.model._system_instruction = content_types.to_content(new_instruction)

    # This is for few-shot free response
    def batch_generate(
        self,
        prompts: List[Tuple[str, Image.Image]],
        top_p: float,
        temperature: float,
        max_tokens: int,
        stop: List[str],
    ) -> List[str]:
        question_type_dict = {
            "Characters on the right side":'leftright',
            "Characters colored red":'color',
            "Characters inside the circle":'circle',}
        system_prompt_dict = {
            "leftright": "Come up with a descriptions for all of the characters on the right side in the image. Your description should be true for all of the characters on the right and none of the characters on the left.",
            "color": "Come up with a descriptions for all of the characters colored red in the image. Your description should be true for all of the characters colored red and none of the characters colored black.",
            "circle": "Come up with a descriptions for all of the characters inside the circle in the image. Your description should be true for all of the characters inside the circle and none of the characters outside the circle."
        }
        generated_texts = []
        for item in prompts:
            groundtruth = item[0]
            #split at the first 'are' occurance
            question_type = question_type_dict[groundtruth.split(' in the image', 1)[0]]
            few_shot_messages = []

            if fewshot_dataset:
                fewshot_typed = [x for x in fewshot_dataset if question_type in x['file_name']]
                for example in fewshot_typed:
                    example_text = example['prompt']
                    example_image = example['images']
                    few_shot_messages.append(example_image)
                    few_shot_messages.append(example_text)

            image = item[1]
            few_shot_messages.append(image)
            system_prompt = system_prompt_dict[question_type]
            try:
                self.update_system_instruction(system_prompt)
                
                response = self.model.generate_content(few_shot_messages)
                if hasattr(response, 'text') and response.text is not None:
                    generated_texts.append(response.text)
                else:
                    print("Response does not contain valid text.")
                    generated_texts.append("None")
            except ValueError as e:
                print(f"Error generating text: {e}")
                generated_texts.append("None")
        return generated_texts 

def main():
    parser = partial_arg_parser()
    parser.add_argument("--model-name", type=str, required=True)
    glyph_pattern_argparser(parser)
    args = parser.parse_args()

    EXTRA_ARGS = ["model_name","fewshot_prompt"]
    super_args = {k: v for (k, v) in vars(args).items() if k not in EXTRA_ARGS}

    generator = GeminiModel(model_name=args.model_name, stop=["\n\n\n"], **super_args)
    global fewshot_dataset
    fewshot_dataset =create_fewshot_dataset(generator.actual_dataset,args.fewshot_prompt)
    generator.generate_all()


if __name__ == "__main__":
    main()

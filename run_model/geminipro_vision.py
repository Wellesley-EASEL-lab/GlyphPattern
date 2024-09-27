import time
from .base import GeneratorBase, partial_arg_parser
from PIL import Image
from io import BytesIO
from typing import List, Tuple
import base64
import google.generativeai as genai
import os
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
        genai.configure(api_key=os.environ["GOOGLE_API_KEY_NEW"])
        self.model = genai.GenerativeModel(model_name="gemini-1.5-pro-001",system_instruction='Reply with only the letter of the correct option.')


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
        generated_texts = []
        for item in prompts:
            text = item[0]
            question_type = question_type_dict[text.split('?\n')[0]]
            few_shot_messages = []

            if fewshot_dataset:
                fewshot_typed = [x for x in fewshot_dataset if question_type in x['file_name']]
                for example in fewshot_typed:
                    example_text = example['prompt']
                    example_image = example['images']
                    example_answer = example['answer']
                    few_shot_messages.append(example_image)
                    message = f"USER: {example_text} \nMODEL: {example_answer}"
                    few_shot_messages.append(message)

            image = item[1]
            few_shot_messages.append(image)
            new_message = f"USER: {text} \nMODEL:"
            few_shot_messages.append(new_message)
            try:
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

    EXTRA_ARGS = ["model_name", "fewshot_prompt"]
    super_args = {k: v for (k, v) in vars(args).items() if k not in EXTRA_ARGS}

    generator = GeminiModel(model_name=args.model_name, stop=["\n\n\n"], **super_args)
    global fewshot_dataset
    fewshot_dataset =create_fewshot_dataset(generator.actual_dataset,args.fewshot_prompt)
    generator.generate_all()


if __name__ == "__main__":
    main()

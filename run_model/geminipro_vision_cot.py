import time
from .base import GeneratorBase, partial_arg_parser
from PIL import Image
from io import BytesIO
from typing import List, Tuple
import base64
import google.generativeai as genai
import os
from .glyphpattern_utils import create_fewshot_dataset,glyph_pattern_argparser

cot_explain = {
    "210_color": "Let's think this through step-by-step. Some of the red shapes don't have loops, so option A is incorrect. Some of the red shapes do not contain a U or a hook, so option B is incorrect. Some of the red shapes do not have a loop at the bottom, so option C is also incorrect. All of the red characters do contain wavy or zigzag lines and none of the black ones do, so option D is a correct description. Since option D is the only pattern that describes all red characters and no black characters, it must be the right answer.",
    "210_circle":"Let's think this through step-by-step. Some of the shapes inside the circle don't have loops, so option A is incorrect. All of the characters inside the circle do contain wavy or zigzag lines and none of the characters outside the circle do, so option B is a correct description. Some of the shapes inside the circle do not have a loop at the bottom, so option C is also incorrect. Some of the shapes inside the circle do not contain a U or a hook, so option D is incorrect. Since option B is the only pattern that describes all characters inside the circle and none of the characters outside the circle, it must be the right answer.",
    "210_leftright":"Let's think this through step-by-step. All of the characters on the right do contain wavy or zigzag lines and none of the characters on the left do, so option A is a correct description. Some of the shapes on the right don't have loops, so option B is incorrect. Some of the shapes on the right do not contain a U or a hook, so option C is incorrect. Some of the shapes on the right do not have a loop at the bottom, so option D is also incorrect. Since option A is the only pattern that describes all characters on the right and none of the characters on the left, it must be the right answer.",
    "298_color": "Let's think this through step-by-step. Almost all of the characters, both red and black, have a vertical straight line, so option A is incorrect. All of the red characters contain at least one enclosed rectangle, so option B is possible. Also, none of the black ones have an enclosed rectangle, so option B is valid. Many of the red characters cannot be drawn with a single stroke, so option C is incorrect. Similarly, some of the red characters do not have an open area facing up, so option D cannot be the answer. Since option B is the only pattern that describes all red characters and no black characters, it must be the right answer.",
    "298_circle":"Let's think this through step-by-step. Many of the characters inside the circle cannot be drawn with a single stroke, so option A is incorrect. Almost all of the characters, both inside and outside the circle, have a vertical straight line, so option B is incorrect. All of the characters inside the circle contain at least one enclosed rectangle, so option C is possible. Also, none of the characters outside the circle ones have an enclosed rectangle, so option C is valid. Some of the characters inside the circle do not have an open area facing up, so option D cannot be the answer. Since option C is the only pattern that describes all characters inside the circle and none of the characters outside the circle, it must be the right answer.",
    "298_leftright":"Let's think this through step-by-step. Almost all of the characters, both on the right and on the left, have a vertical straight line, so option A is incorrect. All of the characters on the right contain at least one enclosed rectangle, so option B is possible. Also, none of the ones on the left have an enclosed rectangle, so option B is valid. Many of the characters on the right cannot be drawn with a single stroke, so option C is incorrect. Similarly, some of the characters on the right do not have an open area facing up, so option D cannot be the answer. Since option B is the only pattern that describes all characters on the right and none of the characters on the left, it must be the right answer.",
    "53_color": "Let's think this through step-by-step. Several of the red characters do not contain an acute angle or diagonal, so option A is incorrect. Not all of the red characters have curves, so option B is incorrect. All of the red characters have two dots, and none of the black shapes have two dots, so option C is a valid rule. Option D is incorrect because many of the red characters do not resemble a bridge with two feet. Therefore, the correct answer is C.",
    "53_circle":"Let's think this through step-by-step. Option A is incorrect because many of the characters inside the circle do not resemble a bridge with two feet. Several of the characters inside the circle do not contain an acute angle or diagonal, so option B is incorrect. Not all of the characters inside the circle have curves, so option C is incorrect. All of the characters inside the circle have two dots, and none of the shapes outside the circle have two dots, so option D is a valid rule. Therefore, the correct answer is D.",
    "53_leftright":"Let's think this through step-by-step. Option A is incorrect because many of the characters on the right do not resemble a bridge with two feet. Several of the characters on the right do not contain an acute angle or diagonal, so option B is incorrect. Not all of the characters on the right have curves, so option C is incorrect. All of the characters on the right have two dots, and none of the shapes on the left have two dots, so option D is a valid rule. Therefore, the correct answer is D."
    }


class GeminiModel(GeneratorBase):
    model_name: str
    model_kwargs: dict
    model: genai.GenerativeModel

    def __init__(self, model_name: str, **super_args):
        super().__init__(**super_args)
        self.model_name = model_name
        self.model = None

    def init_model(self):
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        self.model = genai.GenerativeModel(model_name="gemini-1.5-pro-001",system_instruction="End with Answer: followed by the letter of the correct option")


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
                    cot_answer = cot_explain[example['file_name']]
                    example_answer = cot_answer+" Answer:"+example['answer']
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

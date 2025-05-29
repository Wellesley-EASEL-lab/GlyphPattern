from .base import GeneratorBase, partial_arg_parser
from PIL import Image
from io import BytesIO
from typing import List, Tuple
import base64
from openai import OpenAI
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

def encode_image(image):
    with BytesIO() as output:
        image.save(output, format="PNG")
        return base64.b64encode(output.getvalue()).decode("utf-8")

class GPTModel(GeneratorBase):
    model_name: str
    model_kwargs: dict
    client: OpenAI

    def __init__(self, model_name: str, **super_args):
        super().__init__(**super_args)
        self.model_name = model_name
        self.client = None

    def init_model(self):
        key = os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=key)

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

            few_shot_messages = [
                {
                    "role": "system",
                    "content": "End with Answer: followed by the letter of the correct option",#Let's think step by step.
                }]
            
            if fewshot_dataset:
                #filter out the ones that have questiontype in file_name
                fewshot_typed = [x for x in fewshot_dataset if question_type in x['file_name']]

                for example in fewshot_typed:
                    example_text = example['prompt']
                    example_image = example['images']
                    example_base64_image = encode_image(example_image)
                    example_answer = example['answer']
                    message = {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": example_text},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{example_base64_image}",
                            },
                            },
                        ],
                    }

                    few_shot_messages.append(message)
                    cot_answer = cot_explain[example['file_name']]
                    # print(cot_answer)
                    assistant_message = {
                        "role": "assistant",
                        "content": cot_answer+" Answer:"+example_answer
                    }
                    few_shot_messages.append(assistant_message)

            image = item[1]
            base64_image = encode_image(image)
            new_message = {
                "role": "user",
                "content": [
                    {"type": "text", "text": text},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                        },
                    },
                ],
            }
            few_shot_messages.append(new_message)
            response = self.client.chat.completions.create(
                model="gpt-4o-2024-05-13",
                messages=few_shot_messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                n=1,
                frequency_penalty=0,
                presence_penalty=0,
                stop=stop,
            )
            answer = response.choices[0].message.content
            generated_texts.append(answer)
        return generated_texts
    


def main():
    parser = partial_arg_parser()
    parser.add_argument("--model-name", type=str, required=True)
    glyph_pattern_argparser(parser)
    args = parser.parse_args()

    EXTRA_ARGS = ["model_name", "fewshot_prompt"]
    super_args = {k: v for (k, v) in vars(args).items() if k not in EXTRA_ARGS}

    generator = GPTModel(model_name=args.model_name, stop=["\n\n\n"], **super_args)
    global fewshot_dataset
    fewshot_dataset =create_fewshot_dataset(generator.actual_dataset,args.fewshot_prompt)
    generator.generate_all()


if __name__ == "__main__":
    main()
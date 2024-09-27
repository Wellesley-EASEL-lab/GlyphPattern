from .base import GeneratorBase, partial_arg_parser
from PIL import Image
from io import BytesIO
from typing import List, Tuple
import base64
from openai import OpenAI
import os
from .glyphpattern_utils import create_fewshot_dataset,glyph_pattern_argparser

def encode_image(image):
    with BytesIO() as output:
        image.save(output, format="PNG")
        return base64.b64encode(output.getvalue()).decode("utf-8")

def create_empty_image():
    image = Image.new("RGB", (512,512), (255, 255, 255))  # Create a white image
    return image

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
                    "content": "Reply with only the letter of the correct option.",
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
                    assistant_message = {
                        "role": "assistant",
                        "content": example_answer,
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
                model="gpt-4o",
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
from .base import GeneratorBase, partial_arg_parser
from PIL import Image
from io import BytesIO
from typing import List, Tuple
import base64
from openai import OpenAI
import os
from typing import List, Tuple
import datasets

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

            #filter out the ones that have questiontype in file_name
            fewshot_typed = [x for x in fewshot_dataset if question_type in x['file_name']]
            system_prompt = system_prompt_dict[question_type]
            few_shot_messages = [
                {
                    "role": "system",
                    "content": system_prompt,
                }]
            
            for example in fewshot_typed:
                example_text = example['prompt']
                example_image = example['images']
                example_base64_image = encode_image(example_image)
                message = {
                "role": "user",
                "content": [
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
                    "content": example_text,
                }
                few_shot_messages.append(assistant_message)

            image = item[1]
            base64_image = encode_image(image)
            new_message = {
                "role": "user",
                "content": [
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
    
def create_fewshot_prompt_contents(dataset) -> datasets.Dataset:
    filenames = ['53_color', '210_color', '298_color','53_leftright', '210_leftright','298_leftright','53_circle','210_circle','298_circle']

    fewshot_dataset = dataset.filter(lambda example: example['file_name'] in filenames)

    return fewshot_dataset



def main():
    parser = partial_arg_parser()
    parser.add_argument("--model-name", type=str, required=True)
    args = parser.parse_args()

    EXTRA_ARGS = ["model_name"]
    super_args = {k: v for (k, v) in vars(args).items() if k not in EXTRA_ARGS}

    generator = GPTModel(model_name=args.model_name, stop=["\n\n\n"], **super_args)
    global fewshot_dataset
    fewshot_dataset = generator.create_fewshot_dataset(generator.actual_datase)
    generator.generate_all()


if __name__ == "__main__":
    main()
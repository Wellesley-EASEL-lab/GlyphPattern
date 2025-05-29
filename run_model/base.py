"""
This code generates and saves multiple completions from a dataset of prompts.
It supports the following features:

1. Resumption from errors
2. Comdt storage: each unique completion is stored only once in a json.gz
3. Model framework independent
4. Supports stop tokens.

The code saves completions in a directory that has one file per prompt. Each
file is named "Item_N.json.gz", where N is the index of the prompt in the
dataset. Each file has the following format:

{
    "prompt": PROMPT,
    "temperature": TEMP,
    "top_p": TOP_P,
    "max_tokens": MAX_TOKENS,
    "stop_tokens": [ STOP_TOKEN ... ],
    "extras": EXTRAS,
    "completions": [ { "count": NUMBER, "text": COMPLETION } ... ],
}

Where EXTRAS is a dictionary.
"""

# cspell:ignore tqdm
from typing import List, Tuple, Generator, Optional
from collections import namedtuple
import itertools
from abc import ABC, abstractmethod
import gzip
import json
import argparse
import datasets
from pathlib import Path
from tqdm.auto import tqdm
from tqdm.asyncio import tqdm
from .util import read_json_gz
from .dataset_spec import DatasetSpec
import asyncio

PromptPath = namedtuple("PromptPath", ["prompt", "path", "extras"])

PromptPathCount = namedtuple("PromptPathCount", ["prompt", "path", "count", "extras"])


def stop_at_stop_token(decoded_string, stop_tokens):
    """
    Produces the prefix of decoded_string that ends at the first occurrence of
    a stop_token.

    WARNING: the decoded_string *must not* include the prompt, which may have stop tokens
    itself.
    """
    min_stop_index = len(decoded_string)
    for stop_token in stop_tokens:
        stop_index = decoded_string.find(stop_token)
        if stop_index != -1 and stop_index < min_stop_index:
            min_stop_index = stop_index
    return decoded_string[:min_stop_index]


def partial_arg_parser():
    args = argparse.ArgumentParser()
    args.add_argument("--dataset", type=str, required=True)
    args.add_argument("--output-dir", type=Path, required=True)
    args.add_argument("--completion-limit", type=int, default=200)
    args.add_argument(
        "--batch-size", type=int, default=16, help="Number of completions to batch"
    )
    args.add_argument(
        "--max-tokens",
        type=int,
        required=True,
        help="Maximum number of tokens (prompt and completion)",
    )
    args.add_argument(
        "--top-p",
        type=float,
        default=0.95,
        help="Top-p value for sampling",
    )
    args.add_argument("--temperature", type=float, required=True)
    args.add_argument(
        "--prompt-keys",
        type=str,
        default="prompt",
        help="Comma-separated names of columns in the dataset to use for the prompt",
    )
    args.add_argument(
        "--extra-columns",
        type=str,
        default="",
        help="Comma-separated names of columns in the dataset to include in the completions file",
    )
    return args


def _explode_batch(batch_with_count: List[PromptPathCount]) -> List[PromptPath]:
    """
    Takes a list of PromptPathCount and returns a list of PromptPath, where each
    PromptPathCount is expanded into multiple PromptPath.
    """
    result = []
    for item in batch_with_count:
        for _ in range(item.count):
            result.append(PromptPath(item.prompt, item.path, item.extras))
    return result


def _merge_completions(completions_data, new_completions: List[str]):
    """
    completions_data["completions"] is a list of items

    [ { "count": NUMBER, "text": COMPLETION } ... ]

    We update it in-place.
    """
    completions = completions_data["completions"]
    for completion in new_completions:
        for item in completions:
            if item["text"] == completion:
                item["count"] += 1
                break
        else:
            completions.append({"count": 1, "text": completion})


def _batch_prompts(
    prompts: List[PromptPathCount], batch_size: int
) -> Generator[List[PromptPathCount], None, None]:
    """
    Generates prompts in batches of size batch_size. The batch_size is the
    aggregate count of remaining completions for all prompts in the batch.

    Takes care of splitting a PromptPathCount across batches when needed.
    """
    batch = []
    batch_count = 0
    for prompt in prompts:
        if batch_count == batch_size:
            yield batch
            batch = []
            batch_count = 0
        while prompt.count + batch_count > batch_size:
            # We need to split the prompt across batches.
            take_count = batch_size - batch_count
            drop_count = prompt.count - take_count
            batch.append(PromptPathCount(prompt.prompt, prompt.path, take_count,prompt.extras))
            yield batch
            batch = []
            batch_count = 0
            prompt = PromptPathCount(prompt.prompt, prompt.path, drop_count,prompt.extras)
        batch.append(prompt)
        batch_count += prompt.count

    if len(batch) > 0:
        yield batch


class GeneratorBase(ABC):
    """
    Inherit from this class to generate completions with a particular framework
    or model. The subclass should implement the following methods:

    1. batch_generate: Generates a batch of completions for a list of prompts.

    2. init_model: Initializes the model. This method will be applied exactly
       once after the dataset is loaded. The subclass may load the model in the
       in its __init__ method, and leave this method as `pass`. But, loading the
       model later may help expose data loading errors faster.
    """

    def __init__(
        self,
        dataset: str,
        output_dir: Path,
        completion_limit: int,
        batch_size: int,
        max_tokens: int,
        top_p: float,
        temperature: float,
        stop: List[str],
        prompt_keys: List[str],
        extra_columns: str,
    ):
        self.__dataset = dataset
        self.__output_dir = output_dir.resolve()
        self.__completion_limit = completion_limit
        self.__batch_size = batch_size
        self.__max_tokens__ = max_tokens
        self.__top_p__ = top_p
        self.__temperature__ = temperature
        self.__stop = stop
        prompt_keys = prompt_keys.split(",")
        self.__prompt_keys = prompt_keys
        # "".split(",") == [""] which is not what we want.
        self.__extra_columns = extra_columns.split(",") if len(extra_columns) != 0 else []
        self.actual_dataset = DatasetSpec.from_string(dataset).load()
        

    def __prompts_with_paths(self) -> List[PromptPath]:
        """
        Reads the dataset and returns a list of the prompts in the dataset and
        the full path to the file that should contain the completions for that
        prompt.
        """
        return [
            PromptPath(
                prompt=item[self.__prompt_keys[0]]
                if len(self.__prompt_keys) == 1
                else tuple(item[p] for p in self.__prompt_keys),
                path=self.__output_dir / f"Item_{i}.json.gz",
                extras={key: item[key] for key in self.__extra_columns},
            )
            for i, item in enumerate(self.actual_dataset)
        ]

    def __remaining_prompts(self) -> Tuple[int, List[PromptPathCount]]:
        """
        Returns the number of completions to be generated, and a list of prompts
        that require completions (and their counts).
        """
        num_remaining = 0
        remaining = []
        for prompt, path, extras in self.__prompts_with_paths():
            if not path.exists():
                this_num_remaining = self.__completion_limit
            else:
                completions_data = read_json_gz(path)
                this_num_completed = sum(
                    c["count"] for c in completions_data["completions"]
                )
                this_num_completed = min(this_num_completed, self.__completion_limit)
                this_num_remaining = self.__completion_limit - this_num_completed

            if this_num_remaining > 0:
                num_remaining += this_num_remaining
                remaining.append(
                    PromptPathCount(prompt, path, this_num_remaining, extras)
                )

        return num_remaining, remaining

    def batch_generate(
        self,
        prompts: List[str],
        top_p: float,
        temperature: float,
        max_tokens: int,
        stop: List[str],
    ) -> List[str]:
        raise NotImplementedError("must implement batch_generate to use generate_all")

    @abstractmethod
    def init_model(self):
        pass


    async def generate_single_dynamic(self, prompt: str, top_p: float, temperature: float, max_tokens: int, stop: List[str]) -> str:
        raise NotImplementedError("must implement generate_single_dynamic to use generate_all_dynamic")


    def generate_all(self):
        # Produces an error if __output_dir__ is a file
        self.__output_dir.mkdir(exist_ok=True)

        (num_remaining, prompts_with_paths) = self.__remaining_prompts()
        if num_remaining == 0:
            print("All completions already generated")
            return

        self.init_model()

        num_batches = num_remaining // self.__batch_size
        for batch_with_count in tqdm(
            _batch_prompts(prompts_with_paths, self.__batch_size),
            total=num_batches,
            desc="Batches",
        ):
            batch = _explode_batch(batch_with_count)
            prompts = [item.prompt for item in batch]
            paths = [item.path for item in batch]
            extras = [item.extras for item in batch]
            completions = self.batch_generate(
                prompts,
                self.__top_p__,
                self.__temperature__,
                self.__max_tokens__,
                self.__stop,
            )
            assert len(completions) == len(
                prompts
            ), f"bug in batch_generate: expected {len(prompts)} completions, got {len(completions)}"
            assert type(completions[0]) == str
            groups = sorted(zip(paths, prompts, completions, extras), key=lambda x: x[0])
            for path, group in itertools.groupby(groups, key=lambda x: x[0]):
                group = list(group)
                new_completions = [x[2] for x in group]
                prompts = [x[1] for x in group]
                # assert len(set(prompts)) == 1
                the_prompt = prompts[0]
                the_extras = group[0][3]
                if path.exists():
                    completions_data = read_json_gz(path)
                else:
                    completions_data = {
                        "prompt": the_prompt if type(the_prompt) == str else [p for p in the_prompt if type(p) == str],
                        "temperature": self.__temperature__,
                        "completions": [],
                        "top_p": self.__top_p__,
                        "max_tokens": self.__max_tokens__,
                        "extras": the_extras,
                    }

                _merge_completions(completions_data, new_completions)
                with gzip.open(path, "wt") as f:
                    json.dump(completions_data, f, indent=4)

    async def generate_all_dynamic(self):
        """
        This is a version of generate_all that implements dynamic batching.
        We make an asynchronous request to the model with generate_single_dynamic
        for each prompt. The moment the model returns a completion, we save it
        and issue another concurrent request.
        """
        self.__output_dir.mkdir(exist_ok=True)

        (num_remaining, prompts_with_paths) = self.__remaining_prompts()
        if num_remaining == 0:
            print("All completions already generated")
            return

        self.init_model()

        semaphore = asyncio.Semaphore(self.__batch_size)

        tasks = [self.process_prompt(prompt_path_count, semaphore) for prompt_path_count in _explode_batch(prompts_with_paths)]

        # Use tqdm to wrap the asyncio.gather call
        for f in tqdm.as_completed(tasks, total=len(tasks), desc="Processing Prompts"):
            await f

    async def process_prompt(self, prompt_path: PromptPath, semaphore):
        async with semaphore:
            prompt = prompt_path.prompt
            path = prompt_path.path
            extras = prompt_path.extras
            completion = await self.generate_single_dynamic(
                prompt,
                self.__top_p__,
                self.__temperature__,
                self.__max_tokens__,
                self.__stop,
            )
            # Process the completion (e.g., save it)
            if path.exists():
                completions_data = read_json_gz(path)
            else:
                completions_data = {
                    "prompt": prompt if type(prompt) == str else [p for p in prompt if type(p) == str],
                    "temperature": self.__temperature__,
                    "completions": [],
                    "top_p": self.__top_p__,
                    "max_tokens": self.__max_tokens__,
                    "extras": extras,
                }
            _merge_completions(completions_data, [completion])
            with gzip.open(path, "wt") as f:
                json.dump(completions_data, f, indent=4)



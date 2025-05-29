import json
from pathlib import Path
from typing import List, Generator, TypeVar
import datasets
import abc
import torch
from tqdm.auto import tqdm
import uuid
import csv

T = TypeVar("T")


class _TensorEncoder(json.JSONEncoder):
    """
    A JSON encoder that can serialize torch.Tensor objects.
    """

    def default(self, o):
        if isinstance(o, torch.Tensor):
            return o.tolist()
        return super(_TensorEncoder, self).default(o)


class DatasetSpec(abc.ABC):
    """
    An abstract class that represents a dataset specification. See below for
    concrete implementations for various representations.
    """

    @classmethod
    def from_string(cls, spec: str) -> "DatasetSpec":
        """
        Parse a dataset specification string and return a DatasetSpec.
        """
        kind, dataset_name, *unparsed_flags = spec.split(":")
        flags = {}
        for flag in unparsed_flags:
            # NOTE(arjun): This is terrible code. But, the high-level idea is
            # is that:
            # 1. We interpret a flag as a key value pair, with an "=" separating
            #    the key and value.
            # 2. A flag without a value is interpreted as True
            # 3. For the key limit, we interpret the value as an integer.
            # 4. We treat all other values as strings.
            #
            # There is no validation that the flags make sense for the
            # different types of dataset formats, which is bad.
            if "=" in flag:
                key, value = flag.split("=", 1)
                if key == "limit":
                    value = int(value)
                flags[key] = value
            else:
                flags[flag] = True
        if kind == "csv":
            return _CsvFile(Path(dataset_name).resolve(), **flags)
        elif kind == "jsonl":
            return _JsonlFile(Path(dataset_name).resolve(), **flags)
        elif kind == "disk":
            return _DiskDataset(Path(dataset_name).resolve(), **flags)
        elif kind == "hub":
            if "split" not in flags:
                raise ValueError("split flag is required for hub datasets")
            return _Dataset(dataset_name, **flags)
        else:
            raise ValueError(f"Unknown dataset kind {kind}")

    @abc.abstractmethod
    def save(self, items: Generator[T, None, None]) -> None:
        """
        Save items to the dataset.
        """
        pass

    @abc.abstractmethod
    def load(self) -> datasets.Dataset:
        """
        Load the dataset.
        """
        pass

    def _post_process(self, dataset: datasets.Dataset) -> datasets.Dataset:
        # NOTE(arjun): We shuffle before applying the limit, to ensure that we
        # sample uniformly when shuffling.
        if self._shuffle:
            dataset = dataset.shuffle()
        if self._limit is not None:
            dataset = dataset.select(range(min(self._limit, len(dataset))))
        return dataset

    def __init__(self, limit: int, shuffle: bool):
        self._limit = limit
        self._shuffle = shuffle


class _CsvFile(DatasetSpec):
    """
    A dataset specification that reads and writes to a csv file.
    """

    path: Path

    def __init__(self, path: Path, limit: int = None, shuffle: bool = False):
        super().__init__(limit, shuffle)
        self.path = path

    def save(self, items: Generator[T, None, None]):
        with self.path.open("w") as f:
            writer = csv.writer(f)
            header_written = False
            for item in items:
                if not header_written:
                    writer.writerow(item.keys())
                    header_written = True
                writer.writerow(item.values())

    def load(self) -> datasets.Dataset:
        with self.path.open("r") as f:
            reader = csv.DictReader(f)
            dataset = datasets.Dataset.from_generator(
                _DatasetGeneratorPickleHack(reader)
            )
            return self._post_process(dataset)


class _JsonlFile(DatasetSpec):
    """
    A dataset specification that reads and writes to a jsonl file.
    """

    path: Path

    def __init__(self, path: Path, limit: int = None, shuffle: bool = False):
        super().__init__(limit, shuffle)
        self.path = path

    def save(self, items: Generator[T, None, None]):
        with self.path.open("w") as f:
            for item in tqdm(items):
                json.dump(item, f, cls=_TensorEncoder)
                f.write("\n")

    def load(self) -> datasets.Dataset:
        dataset = datasets.Dataset.from_json(str(self.path))
        return self._post_process(dataset)


class _Dataset(DatasetSpec):
    """
    A dataset specification that reads and writes to a Hugging Face dataset.
    using `datasets.load_dataset` and `Dataset.push_to_hub`.
    """

    dataset_name: str
    flags: dict

    # NOTE(arjun): It is very important that the named arguments below *not*
    # overlap with the named arguments of `datasets.load_dataset`. For example,
    # we use `dataset_name` instead of `name`, since the latter is used to
    # refer to the "configuration" of a dataset on the Hugging Face Hub.
    def __init__(
        self,
        dataset_name: str,
        push_to_hub: bool = False,
        private: bool = False,
        limit: int = None,
        shuffle: bool = False,
        **flags,
    ):
        super().__init__(limit, shuffle)
        self.dataset_name = dataset_name
        self.flags = flags
        self.push_to_hub = push_to_hub
        self.private = private

    def save(self, items: Generator[T, None, None]):
        items = _DatasetGeneratorPickleHack(items)
        ds = datasets.Dataset.from_generator(items)
        # Addresses the inconsistency in the Hugging Face Hub API
        # between `datasets.push_to_hub` and `datasets.load_dataset`.
        flags = {**self.flags}
        if flags["name"] is not None:
            flags["config_name"] = flags["name"]
            del flags["name"]

        if self.push_to_hub:
            ds.push_to_hub(repo_id=self.dataset_name, **flags, private=self.private)

    def load(self) -> datasets.Dataset:
        dataset = datasets.load_dataset(self.dataset_name, **self.flags)
        return self._post_process(dataset)


class _DiskDataset(DatasetSpec):
    """
    A dataset specification that reads and writes to a directory on disk
    using `Dataset.save_to_disk` and `Dataset.load_from_disk`.
    """

    path: Path

    def __init__(self, path: Path, limit: int = None, shuffle: bool = False):
        super().__init__(limit, shuffle)
        self.path = path

    def save(self, items: Generator[T, None, None]):
        items = _DatasetGeneratorPickleHack(items)
        ds = datasets.Dataset.from_generator(items)
        ds.save_to_disk(str(self.path))

    def load(self) -> datasets.Dataset:
        dataset = datasets.Dataset.load_from_disk(str(self.path))
        return self._post_process(dataset)


# https://github.com/huggingface/datasets/issues/6194#issuecomment-1708080653
class _DatasetGeneratorPickleHack:
    def __init__(self, generator, generator_id=None):
        self.generator = generator
        self.generator_id = (
            generator_id if generator_id is not None else str(uuid.uuid4())
        )

    def __call__(self, *args, **kwargs):
        return self.generator

    def __reduce__(self):
        return (_DatasetGeneratorPickleHack_raise, (self.generator_id,))


def _DatasetGeneratorPickleHack_raise(*args, **kwargs):
    raise AssertionError("cannot actually unpickle _DatasetGeneratorPickleHack!")

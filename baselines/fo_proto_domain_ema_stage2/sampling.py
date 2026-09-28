"""Local episode adapter; shared Task, datasets and loaders stay untouched."""
from dataclasses import dataclass
from itertools import count, zip_longest
from pathlib import Path

import numpy as np
import torch
from sklearn.utils import check_random_state

from cdmetadl.ingestion.data_generator import CompetitionDataLoader, Task
from .data import context_for_task


@dataclass(frozen=True)
class Episode:
    task: Task
    context: dict
    reference_sampler_verified: bool = False

    def __getattr__(self, name):
        return getattr(self.task, name)


class EpisodeLoader:
    """Same sampling order as CompetitionDataLoader, with local identities.

    The original loader supplies configuration validation and way/shot logic.
    A local sampler supports an uninterrupted, unbounded training stream and
    records indices at selection time. Finite validation/test streams are
    checked against the original loader task by task before reaching a model.
    No source instance, class, module or Task is patched or given attributes.
    """
    def __init__(self, datasets, config, seed, test_generator=False):
        self.reference_loader = CompetitionDataLoader(
            datasets, config, seed, test_generator=test_generator)
        self.datasets, self.seed, self.test_generator = datasets, seed, test_generator
        self.parity_tasks = 0

    def _sample(self, rng, dataset):
        loader = self.reference_loader
        ways, shots = loader.prepare_task_config(
            rng, len(dataset.idx_per_label), dataset.min_examples_per_class - loader.query_size)
        support_size = ways * shots
        examples_per_class = shots + loader.query_size
        classes = rng.permutation(len(dataset.idx_per_label))[:ways]
        indices = np.stack([rng.choice(dataset.idx_per_label[c], examples_per_class, replace=False)
                            for c in classes]).T.reshape(-1)
        images, originals = [], []
        for index in indices:
            image, label = dataset[index]
            images.append(image)
            originals.append(label)
        images, originals = torch.stack(images), torch.stack(originals).long()
        labels = torch.arange(ways).repeat(examples_per_class).long()
        original_classes = originals.numpy()[:ways]
        support_order = rng.permutation(support_size)
        query_order = rng.permutation(len(images) - support_size)
        support = (images[:support_size][support_order], labels[:support_size][support_order],
                   originals[:support_size][support_order])
        query = (images[support_size:][query_order], labels[support_size:][query_order],
                 originals[support_size:][query_order])
        task = Task(ways, shots, support, query, original_classes, dataset.name)
        # Domain is read from the actual Task.dataset, never its stream position.
        context = context_for_task(task)
        context["episode"] = dict(
            classes=classes.tolist(),
            support_indices=indices[:support_size][support_order].tolist(),
            query_indices=indices[support_size:][query_order].tolist(),
            support_files=[Path(dataset.img_paths[int(i)]).name
                           for i in indices[:support_size][support_order]],
            query_files=[Path(dataset.img_paths[int(i)]).name
                         for i in indices[support_size:][query_order]])
        return Episode(task, context)

    def _episodes(self, num_tasks):
        rng = check_random_state(self.seed)
        if self.test_generator:
            if num_tasks is None:
                raise ValueError("Test tasks per dataset must be finite")
            for dataset in self.datasets:
                for _ in range(num_tasks):
                    yield self._sample(rng, dataset)
        else:
            for _ in (count() if num_tasks is None else range(num_tasks)):
                dataset = self.datasets[rng.randint(0, len(self.datasets))]
                yield self._sample(rng, dataset)

    @staticmethod
    def _compare(local, original):
        if (local.dataset, local.num_ways, local.num_shots) != (
                original.dataset, original.num_ways, original.num_shots):
            raise AssertionError("Local/reference dataset, ways or shots differ")
        if not np.array_equal(local.original_class_idx, original.original_class_idx):
            raise AssertionError("Local/reference class order differs")
        for actual, expected in zip((*local.support_set, *local.query_set),
                                    (*original.support_set, *original.query_set)):
            if not torch.equal(actual, expected):
                raise AssertionError("Local/reference ordered images or labels differ")

    def generator(self, num_tasks):
        if num_tasks is None:
            # A single persistent RNG for quota sampling. No range(None) is
            # passed to the original generator and no chunk restarts occur.
            yield from self._episodes(None)
            return
        for local, original in zip_longest(
                self._episodes(num_tasks), self.reference_loader.generator(num_tasks)):
            if local is None or original is None:
                raise AssertionError("Local/reference episode count differs")
            self._compare(local, original)
            self.parity_tasks += 1
            # Use the original loader's real Task after checking identities.
            context = context_for_task(original)
            context["episode"] = local.context["episode"]
            yield Episode(original, context, reference_sampler_verified=True)

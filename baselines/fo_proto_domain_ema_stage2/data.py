"""Explicit taxonomy, FEEDBACK validation and reproducible episode identity."""
import hashlib
import json
from pathlib import Path

from cdmetadl.ingestion.image_dataset import create_datasets
from cdmetadl.helpers.general_helpers import prepare_datasets_information

TAXONOMY = json.loads(Path(__file__).with_name("domains.json").read_text())
DOMAINS = TAXONOMY["domains"]
DOMAIN_INDEX = {domain: i for i, domain in enumerate(DOMAINS)}
ALIASES = {}
for (domain, ids), prefix in zip(DOMAINS.items(), TAXONOMY["domain_ids"]):
    for identifier in ids:
        for alias in (identifier, identifier + "_Mini", prefix + "." + identifier):
            if alias in ALIASES:
                raise ValueError(f"Conflicting dataset alias: {alias}")
            ALIASES[alias] = identifier
DATASET_DOMAIN = {identifier: domain for domain, ids in DOMAINS.items() for identifier in ids}
TEST_CONFIG = dict(N=None, min_N=2, max_N=20, k=None, min_k=1, max_k=20,
                   query_images_per_class=20)


def canonical(name):
    # Paths may have a set directory prefix. No case folding/fuzzy matching.
    leaf = str(name).replace("\\", "/").rstrip("/").split("/")[-1]
    if leaf not in ALIASES:
        raise ValueError(f"Unknown dataset {name!r}; add a documented exact alias")
    return ALIASES[leaf]


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def validate_split(infos):
    manifest = {}
    for split, info in zip(("train", "validation", "test"), infos):
        rows = []
        for name, (_, _, image_path, labels_path) in info.items():
            identifier = canonical(name)
            if canonical(Path(image_path).parent.name) != identifier:
                raise ValueError(f"Folder and split ID disagree: {name}")
            metadata_path = Path(labels_path).with_name("info.json")
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            # A descriptive dataset_name is not a domain. Check it only when
            # it is one of the explicit ID aliases; never require a domain key.
            declared = metadata.get("dataset_name")
            if declared in ALIASES and ALIASES[declared] != identifier:
                raise ValueError(f"Metadata and folder ID disagree: {name}")
            rows.append(dict(name=name, dataset_id=identifier, domain=DATASET_DOMAIN[identifier],
                             labels_sha256=file_hash(labels_path), info_sha256=file_hash(metadata_path)))
        ids = [row["dataset_id"] for row in rows]
        if len(set(ids)) != len(ids):
            raise ValueError(f"Dataset aliases collide in {split}")
        manifest[split] = rows
    train = {r["dataset_id"] for r in manifest["train"]}
    valid = {r["dataset_id"] for r in manifest["validation"]}
    test = {r["dataset_id"] for r in manifest["test"]}
    if (len(train) != 7 or len(valid) != 3 or train & valid
            or train | valid != {ids[0] for ids in DOMAINS.values()}
            or test != {ids[1] for ids in DOMAINS.values()}):
        raise ValueError("Requires FEEDBACK: Set-0 7 train/3 validation; all Set-1 test")
    return manifest


def context_for_task(task):
    if not isinstance(task.dataset, str) or not task.dataset:
        raise ValueError("Real Task.dataset metadata is required; no inferred routing")
    identifier = canonical(task.dataset)
    return dict(dataset_id=identifier, domain=DATASET_DOMAIN[identifier])


def loaders(root, source_config, options, image_size=128):
    from .sampling import EpisodeLoader
    infos = prepare_datasets_information(root, 3, options["data_seed"], False)
    manifest = validate_split(infos)
    result = []
    for info, cfg, seed, test in zip(infos,
            (source_config["train_config"], source_config["valid_config"], TEST_CONFIG),
            (options["stage2_seed"], options["validation_seed"], options["test_seed"]),
            (False, False, True)):
        result.append(EpisodeLoader(create_datasets(info, image_size), cfg, seed, test_generator=test))
    return result, manifest


def episode_record(task):
    if not getattr(task, "context", None) or "episode" not in task.context:
        raise ValueError("Episode identity/context is missing; use the local EpisodeLoader")
    if any(task.context[key] != value for key, value in context_for_task(task).items()):
        raise ValueError("Episode context disagrees with real Task.dataset")
    record = dict(task.context, ways=task.num_ways, shots=task.num_shots)
    # Hash realized tensors as well as ordered paths/classes: includes any
    # future stochastic augmentation, not just its sampling seed.
    digest = hashlib.sha256()
    for tensor in (*task.support_set, *task.query_set):
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(str((array.dtype, array.shape)).encode())
        digest.update(array.tobytes())
    record["tensor_sha256"] = digest.hexdigest()
    record["task_id"] = json_hash(record)
    return record

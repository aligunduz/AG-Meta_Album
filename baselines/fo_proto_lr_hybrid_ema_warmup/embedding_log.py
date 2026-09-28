"""Bounded float32 chunks and JSONL metadata; no model/data RNG use."""
import hashlib
import json
from pathlib import Path
import numpy as np


def plain(value):
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def measures(v):
    v = {k: np.asarray(x, dtype=np.float64) for k, x in v.items()}
    result = {k + "_norm": float(np.linalg.norm(x)) for k, x in v.items()}
    for a, b in (("z_task", "z_ema_before"), ("z_task", "z_hybrid"), ("z_hybrid", "z_ema_before")):
        na, nb = result[a+"_norm"], result[b+"_norm"]
        result[a+"__"+b+"_l2"] = float(np.linalg.norm(v[a]-v[b]))
        result[a+"__"+b+"_cosine"] = float(np.dot(v[a], v[b])/(na*nb)) if min(na,nb)>1e-12 else None
    n = result["z_ema_before_norm"]
    result["task_ema_relative_distance"] = result["z_task__z_ema_before_l2"]/n if n>1e-12 else None
    result["invalid_metric_count"] = sum(x is None for x in result.values())
    return result


class EpisodeWriter:
    def __init__(self, directory, config):
        self.directory = Path(directory)
        self.config = config
        self.enabled = config["embedding_logging"]["enabled"]
        self.chunk_size = config["embedding_logging"]["chunk_size"]
        if type(self.chunk_size) is not int or self.chunk_size < 1:
            raise ValueError("chunk_size must be a positive integer")
        self.pending = []
        self.sums, self.counts = {}, {}
        self.next_id = 0
        if not self.enabled:
            return
        self.directory.mkdir(parents=True, exist_ok=True)
        chunks = sorted(self.directory.glob("chunk-*.jsonl"))
        if chunks:
            with chunks[-1].open(encoding="utf-8") as h:
                for line in h:
                    self.next_id = max(self.next_id, json.loads(line)["record_id"]+1)
        meta = dict(run=config.get("run_name", self.directory.parent.parent.name),
            data_seed=config.get("data_seed"), model_seed=config.get("model_seed"),
            encoder_trainable=True, embedding_dim=512,
            embedding_source="hybrid model initial support forward with its own current encoder weights",
            conditioning_detached=True, config=config)
        (self.directory/"run.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    def append(self, row, vectors):
        if not self.enabled:
            return
        row = dict(row, record_id=self.next_id, **measures(vectors))
        self.next_id += 1
        self.pending.append((row, {k: np.asarray(v, dtype=np.float32).copy() for k,v in vectors.items()}))
        for key, value in row.items():
            if key.endswith(("_norm", "_l2", "_cosine")) or key in ("task_ema_relative_distance", "invalid_metric_count"):
                if value is not None:
                    self.sums[key] = self.sums.get(key,0)+value
                    self.counts[key] = self.counts.get(key,0)+1
        if len(self.pending) >= self.chunk_size:
            self.flush()

    def summary(self):
        result = {"embedding/"+k: v/self.counts[k] for k,v in self.sums.items()}
        result["embedding/invalid_metric_total"] = self.sums.get("invalid_metric_count", 0)
        self.sums, self.counts = {}, {}
        return result

    def flush(self):
        if not self.pending:
            return
        stem = "chunk-%012d" % self.pending[0][0]["record_id"]
        arrays = {k: np.stack([v[k] for _,v in self.pending]) for k in self.pending[0][1]}
        arrays["record_id"] = np.array([r["record_id"] for r,_ in self.pending], dtype=np.int64)
        np.savez_compressed(self.directory/(stem+".npz"), **arrays)
        with (self.directory/(stem+".jsonl")).open("w", encoding="utf-8") as h:
            for index,(row,_) in enumerate(self.pending):
                h.write(json.dumps(dict(row, vector_file=stem+".npz", vector_row=index), allow_nan=False)+"\n")
        self.pending.clear()


def record_episode(writer, transport, task, phase, step, validation_round, out, loss, ordinal=None):
    if not writer.enabled:
        return
    observation = dict(transport.observation)
    vectors = observation.pop("vectors")
    support_counts = np.bincount(task.support_set[1].cpu().numpy(), minlength=task.num_ways).tolist()
    query_counts = np.bincount(task.query_set[1].cpu().numpy(), minlength=task.num_ways).tolist()
    sid = plain(getattr(task, "support_ids", None))
    qid = plain(getattr(task, "query_ids", None))
    dataset = getattr(task, "dataset", None)
    fingerprint = None
    if sid is not None and qid is not None:
        fingerprint = hashlib.sha256(json.dumps([dataset,sid,qid], sort_keys=True).encode()).hexdigest()
    corrections = getattr(transport, "episode_corrections", [])
    correction_stats = {}
    for i,key in enumerate(("low_rank_correction_norm", "original_gradient_norm", "correction_to_gradient_ratio")):
        values = [v[i] for v in corrections if v[i] is not None]
        correction_stats[key] = float(np.mean(values)) if values else None
    row = dict(observation, **correction_stats,
        correction_invalid_count=sum(v[2] is None for v in corrections),
        run=writer.config.get("run_name", writer.directory.parent.parent.name), phase=phase,
        training_step=step, checkpoint="step-%s" % step, validation_round=validation_round,
        task_ordinal=ordinal if ordinal is not None else writer.next_id,
        data_seed=writer.config.get("data_seed"), model_seed=writer.config.get("model_seed"),
        episode_seed=getattr(task,"episode_seed",None), dataset_name=dataset,
        dataset_id=getattr(task,"dataset_id",None), domain_name=getattr(task,"domain",None),
        domain_id=getattr(task,"domain_id",None), way=task.num_ways, shot=task.num_shots,
        support_counts=support_counts, query_counts=query_counts, query_count=sum(query_counts),
        class_ids=plain(getattr(task,"original_class_idx",None)), support_ids=sid, query_ids=qid,
        episode_fingerprint=fingerprint, query_loss=loss.detach().item() if loss is not None else None,
        query_accuracy=(out.detach().argmax(1).cpu()==task.query_set[1].cpu()).float().mean().item())
    writer.append(row, vectors)

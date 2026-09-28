"""Offline descriptive analysis. Bounded reservoir per phase/checkpoint/dataset/domain."""
import argparse
import json
from collections import defaultdict
from pathlib import Path
import numpy as np


def distribution(values):
    a = np.asarray([x for x in values if x is not None], dtype=float)
    return dict(count=len(a), mean=float(a.mean()) if len(a) else None,
                quantiles=np.quantile(a,[0,.25,.5,.75,1]).tolist() if len(a) else [])


def analyze(directory, output, seed=98, max_tasks=512, max_pairs=10000):
    if max_tasks < 2 or max_pairs < 1:
        raise ValueError("max_tasks >= 2 and max_pairs >= 1 required")
    rng = np.random.default_rng(seed)
    groups = {}
    strata = defaultdict(list)
    movement = defaultdict(list)
    previous = {}
    for file in sorted(Path(directory).glob("chunk-*.jsonl")):
        rows = [json.loads(line) for line in file.read_text(encoding="utf-8").splitlines()]
        with np.load(file.with_suffix(".npz")) as arrays:
            for row in rows:
                index = row["vector_row"]
                if arrays["record_id"][index] != row["record_id"]:
                    raise ValueError("Metadata/vector record mismatch")
                # Training uses bounded stage bins; validation/test retain exact checkpoints.
                stage = "warmup" if row["warmup"] else "post_warmup"
                checkpoint = row["checkpoint"] if row["phase"] != "train" else "%s/bin-%s" % (stage, (row["training_step"]-1)//5000)
                key = (row["run"],row["phase"],checkpoint,row["dataset_name"] if row["dataset_name"] is not None else row.get("dataset_id"),row["domain_name"] if row["domain_name"] is not None else row.get("domain_id"))
                z = arrays["z_task"][index].astype(np.float64)
                g = groups.setdefault(key, dict(n=0, sum=np.zeros_like(z), square=0., norms=0., sample=[]))
                g["n"] += 1; g["sum"] += z; g["square"] += float(z@z); g["norms"] += float(np.linalg.norm(z))
                item = (z.copy(), row)
                if len(g["sample"]) < max_tasks:
                    g["sample"].append(item)
                else:
                    j = rng.integers(g["n"])
                    if j < max_tasks:
                        g["sample"][j] = item
                if row["phase"] == "train":
                    ema = arrays["z_ema_before"][index].copy()
                    run = row["run"]
                    if run in previous and row["ema_initialized"]:
                        movement[(run,stage)].append(float(np.linalg.norm(ema-previous[run])))
                    previous[run] = ema
    report = dict(seed=seed, max_tasks_per_group=max_tasks, max_pairs=max_pairs,
        note="Encoder is trained; temporal distances also include representation-space changes. Descriptive associations are not causal. Train groups use 5000-task bins; evaluation checkpoints and phases are separate. Unknown domains stay null.", groups=[], pairwise=[], centers=[], ema_movement=[])
    partitions = defaultdict(list)
    for key,g in groups.items():
        center = g["sum"]/g["n"]
        report["groups"].append(dict(key=key,count=g["n"],center=center.tolist(),mean_norm=g["norms"]/g["n"], rms_spread=float(np.sqrt(max(0,g["square"]/g["n"]-center@center))),sample_count=len(g["sample"])))
        partitions[key[:3]].append((key,g,center))
        for z,row in g["sample"]:
            strata[(key,row["shot"])].append(row)
    for partition,entries in partitions.items():
        centers = np.stack([c for _,_,c in entries])
        report["centers"].append(dict(partition=partition, labels=[k[3:] for k,_,_ in entries], l2=np.linalg.norm(centers[:,None]-centers[None,:],axis=2).tolist()))
        for i,(ka,ga,_) in enumerate(entries):
            for kb,gb,_ in entries[i:]:
                sa,sb=ga["sample"],gb["sample"]
                l2,cos=[],[]
                if ga is gb and len(sa)<2:
                    continue
                for _ in range(min(max_pairs,len(sa)*len(sb))):
                    a=int(rng.integers(len(sa)))
                    b=int(rng.integers(len(sb)-1 if ga is gb else len(sb)))
                    if ga is gb and b>=a:
                        b+=1
                    x,y=sa[a][0],sb[b][0]; nx,ny=np.linalg.norm(x),np.linalg.norm(y)
                    l2.append(float(np.linalg.norm(x-y))); cos.append(float(x@y/(nx*ny)) if min(nx,ny)>1e-12 else None)
                report["pairwise"].append(dict(a=ka,b=kb,kind="within_dataset" if ka[3]==kb[3] else "between_dataset",l2=distribution(l2),cosine=distribution(cos),l2_samples=l2,cosine_samples=cos,invalid_cosine=sum(v is None for v in cos)))
    report["task_ema_by_shot"]=[]
    for (key,shot),rows in strata.items():
        distances=[r["z_task__z_ema_before_l2"] for r in rows]
        correlations={}
        for metric in ("query_accuracy","query_loss","support_accuracy","support_loss"):
            pairs=np.array([(r["z_task__z_ema_before_l2"],r[metric]) for r in rows if r.get(metric) is not None])
            correlations[metric]=float(np.corrcoef(pairs.T)[0,1]) if len(pairs)>1 and np.all(pairs.std(0)>1e-12) else None
        report["task_ema_by_shot"].append(dict(key=key,shot=shot,distance=distribution(distances),samples=distances,pearson=correlations))
    for key,values in movement.items():
        report["ema_movement"].append(dict(key=key,consecutive_l2=distribution(values),samples=values))
    Path(output).write_text(json.dumps(report,indent=2,allow_nan=False),encoding="utf-8")
    return report


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory"); parser.add_argument("--output",default="embedding_analysis.json")
    parser.add_argument("--seed",type=int,default=98)
    parser.add_argument("--max-tasks",type=int,default=512)
    parser.add_argument("--max-pairs",type=int,default=10000)
    args=parser.parse_args()
    analyze(args.directory,args.output,args.seed,args.max_tasks,args.max_pairs)

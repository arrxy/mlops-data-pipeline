"""
Q2.6 batch-compiler
Build (query, positive_id, negative_id, query_type) training tuples from the dataset.
Split by image_id hash: 70% train / 15% val / 15% test.
Assert zero overlap between train and test image IDs (leakage check).
Upload train.jsonl, val.jsonl, test.jsonl, manifest.json to Swift.

Dataset structure (unified_dataset.json):
  image_id: str
  queries:
    raw:     list[str]   - multiple raw queries per image
    tags:    list[str]   - descriptive tags
    caption: str         - single caption
  split: str
"""

import hashlib
import io
import json
import os
import random
from datetime import datetime, timezone

import boto3
from botocore.client import Config
from huggingface_hub import hf_hub_download

BUCKET = "ObjStore_proj24"
PREFIX = "datasets/v1"
S3_ENDPOINT = "https://chi.tacc.chameleoncloud.org:7480"
HF_REPO = "ar10067/flickr30k-images-CFQ"

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
# TEST_RATIO = 0.15 (implicit)


def image_id_split(image_id: str) -> str:
    """Deterministically assign an image to train/val/test by hashing its ID."""
    h = int(hashlib.md5(str(image_id).encode()).hexdigest(), 16)
    bucket = (h % 100) / 100.0
    if bucket < TRAIN_RATIO:
        return "train"
    elif bucket < TRAIN_RATIO + VAL_RATIO:
        return "val"
    else:
        return "test"


def get_s3_client():
    return boto3.client(
        "s3",
        endpoint_url=S3_ENDPOINT,
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
        config=Config(signature_version="s3"),
    )


def iter_queries(record):
    """
    Yield (query_text, query_type) pairs for a single record.
      raw:     each string in the list
      tags:    joined into a single comma-separated query
      caption: the string directly
    """
    queries = record.get("queries", {})

    for q in queries.get("raw", []):
        if q and q.strip():
            yield q.strip(), "raw"

    tags = [t.strip() for t in queries.get("tags", []) if t and t.strip()]
    if tags:
        yield ", ".join(tags), "tags"

    caption = queries.get("caption", "")
    if caption and caption.strip():
        yield caption.strip(), "caption"


def main():
    print(f"Downloading unified_dataset.json from {HF_REPO}...")
    local_path = hf_hub_download(
        repo_id=HF_REPO,
        filename="unified_dataset.json",
        repo_type="dataset",
        token=os.environ.get("HF_TOKEN"),
    )
    with open(local_path) as f:
        records = json.load(f)
    print(f"Loaded {len(records):,} records.")

    all_image_ids = [r["image_id"] for r in records]

    splits: dict[str, list] = {"train": [], "val": [], "test": []}

    print("Building training tuples...")
    for record in records:
        pos_id = record["image_id"]
        split = image_id_split(pos_id)

        for query_text, query_type in iter_queries(record):
            # Sample a negative image different from the positive
            neg_id = pos_id
            while neg_id == pos_id:
                neg_id = random.choice(all_image_ids)

            splits[split].append({
                "query": query_text,
                "positive_id": pos_id,
                "negative_id": neg_id,
                "query_type": query_type,
            })

    train_ids = {t["positive_id"] for t in splits["train"]}
    test_ids = {t["positive_id"] for t in splits["test"]}
    overlap = train_ids & test_ids
    assert len(overlap) == 0, (
        f"Leakage detected: {len(overlap)} image IDs appear in both train and test."
    )
    print("Leakage check passed: zero overlap between train and test image IDs.")

    for split_name, tuples in splits.items():
        print(f"  {split_name}: {len(tuples):,} tuples")

    s3 = get_s3_client()

    for split_name, tuples in splits.items():
        buf = io.BytesIO()
        for t in tuples:
            buf.write((json.dumps(t) + "\n").encode("utf-8"))
        buf.seek(0)
        key = f"{PREFIX}/{split_name}.jsonl"
        print(f"Uploading s3://{BUCKET}/{key}...")
        s3.put_object(Bucket=BUCKET, Key=key, Body=buf.read())

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": HF_REPO,
        "split_ratios": {
            "train": TRAIN_RATIO,
            "val": VAL_RATIO,
            "test": round(1 - TRAIN_RATIO - VAL_RATIO, 2),
        },
        "counts": {k: len(v) for k, v in splits.items()},
        "leakage_check": "passed",
    }
    manifest_key = f"{PREFIX}/manifest.json"
    print(f"Uploading s3://{BUCKET}/{manifest_key}...")
    s3.put_object(
        Bucket=BUCKET,
        Key=manifest_key,
        Body=json.dumps(manifest, indent=2).encode("utf-8"),
    )
    print("Done.")


if __name__ == "__main__":
    main()

"""
Q2.4 data-generator
Simulate user search traffic using queries from the dataset.
Generates (query, ranked_ids, clicked_id, timestamp) interaction events
for 120 seconds at ~2 RPS, then uploads interactions.jsonl to Swift.

Dataset structure:
  image_id: str
  queries:
    raw:     list[str]   - multiple raw queries per image
    tags:    list[str]   - descriptive tags
    caption: str         - single caption
  split: str
"""

import json
import os
import io
import random
import time
from datetime import datetime, timezone

import boto3
from botocore.client import Config
from huggingface_hub import hf_hub_download

BUCKET = "ObjStore_proj24"
PREFIX = "interactions"
S3_ENDPOINT = "https://chi.tacc.chameleoncloud.org:7480"
HF_REPO = "ar10067/flickr30k-images-CFQ"

DURATION_SECONDS = 120
TARGET_RPS = 2
RANKED_LIST_SIZE = 10


def get_s3_client():
    return boto3.client(
        "s3",
        endpoint_url=S3_ENDPOINT,
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
        config=Config(signature_version="s3"),
    )


def build_query_pool(records):
    """
    Flatten all (query_text, image_id, query_type) entries.
    raw  → pick one string from the list per event
    tags → join into a single query string
    caption → use the string directly
    """
    pool = []
    for record in records:
        image_id = record["image_id"]
        queries = record.get("queries", {})

        # raw: list of natural-language queries
        for q in queries.get("raw", []):
            if q and q.strip():
                pool.append({"query": q.strip(), "image_id": image_id, "query_type": "raw"})

        # tags: list of tag strings → join as a tag-style query
        tags = queries.get("tags", [])
        if tags:
            tag_query = ", ".join(t.strip() for t in tags if t.strip())
            if tag_query:
                pool.append({"query": tag_query, "image_id": image_id, "query_type": "tags"})

        # caption: single string
        caption = queries.get("caption", "")
        if caption and caption.strip():
            pool.append({"query": caption.strip(), "image_id": image_id, "query_type": "caption"})

    return pool


def simulate_ranked_list(positive_id, all_ids, k=RANKED_LIST_SIZE):
    candidates = random.sample([i for i in all_ids if i != positive_id], min(k - 1, len(all_ids) - 1))
    ranked = candidates + [positive_id]
    random.shuffle(ranked)
    return ranked


def main():
    print(f"Downloading unified_dataset.json from {HF_REPO}...")
    local_path = hf_hub_download(
        repo_id=HF_REPO,
        filename="unified_dataset.json",
        repo_type="dataset",
    )
    with open(local_path) as f:
        records = json.load(f)
    print(f"Loaded {len(records):,} records.")

    all_image_ids = [r["image_id"] for r in records]
    query_pool = build_query_pool(records)
    print(f"Query pool size: {len(query_pool):,}")

    events = []
    interval = 1.0 / TARGET_RPS
    start_time = time.monotonic()
    deadline = start_time + DURATION_SECONDS

    print(f"Simulating traffic for {DURATION_SECONDS}s at ~{TARGET_RPS} RPS...")
    while time.monotonic() < deadline:
        tick_start = time.monotonic()

        entry = random.choice(query_pool)
        ranked_ids = simulate_ranked_list(entry["image_id"], all_image_ids)
        # Click probability biased toward top of ranked list
        click_weights = [1.0 / (i + 1) for i in range(len(ranked_ids))]
        clicked_id = random.choices(ranked_ids, weights=click_weights, k=1)[0]

        events.append({
            "query": entry["query"],
            "query_type": entry["query_type"],
            "ranked_ids": ranked_ids,
            "clicked_id": clicked_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

        elapsed = time.monotonic() - tick_start
        sleep_time = interval - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)

    print(f"Generated {len(events):,} interaction events.")

    buf = io.BytesIO()
    for event in events:
        buf.write((json.dumps(event) + "\n").encode("utf-8"))
    buf.seek(0)

    s3 = get_s3_client()
    key = f"{PREFIX}/interactions.jsonl"
    print(f"Uploading s3://{BUCKET}/{key}...")
    s3.put_object(Bucket=BUCKET, Key=key, Body=buf.read())
    print("Done.")


if __name__ == "__main__":
    main()

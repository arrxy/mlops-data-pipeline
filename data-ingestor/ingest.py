"""
Q2.3 data-ingestor
Pull unified_dataset.json from HuggingFace (ar10067/flickr30k-images-CFQ)
and upload raw data to Swift at ObjStore_proj24/raw/v1/.
Writes a completion marker ingest_done.json.
"""

import json
import os
import io
import zipfile
from datetime import datetime, timezone

import boto3
from botocore.client import Config
from huggingface_hub import hf_hub_download

BUCKET = "ObjStore_proj24"
PREFIX = "raw/v1"
S3_ENDPOINT = "https://chi.tacc.chameleoncloud.org:7480"
HF_REPO = "ar10067/flickr30k-images-CFQ"


def get_s3_client():
    return boto3.client(
        "s3",
        endpoint_url=S3_ENDPOINT,
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
        config=Config(signature_version="s3"),
    )


def main():
    print(f"Downloading unified_dataset.json from {HF_REPO}...")
    local_path = hf_hub_download(
        repo_id=HF_REPO,
        filename="unified_dataset.json",
        repo_type="dataset",
        token=os.environ.get("HF_TOKEN"),
    )
    print(f"Downloaded to {local_path}")

    with open(local_path, "rb") as f:
        raw_bytes = f.read()

    records = json.loads(raw_bytes)
    print(f"Loaded {len(records):,} records.")

    # Re-serialise as JSONL for easier downstream streaming
    buf = io.BytesIO()
    for record in records:
        buf.write((json.dumps(record) + "\n").encode("utf-8"))
    jsonl_bytes = buf.getvalue()

    print(f"Downloading flickr30k-images.zip from {HF_REPO}...")
    zip_path = hf_hub_download(
        repo_id=HF_REPO,
        filename="flickr30k-images.zip",
        repo_type="dataset",
        token=os.environ.get("HF_TOKEN"),
    )
    print(f"Downloaded to {zip_path}")

    s3 = get_s3_client()

    # Upload original JSON
    json_key = f"{PREFIX}/unified_dataset.json"
    print(f"Uploading s3://{BUCKET}/{json_key} ({len(raw_bytes):,} bytes)...")
    s3.put_object(Bucket=BUCKET, Key=json_key, Body=raw_bytes)

    # Upload JSONL version
    jsonl_key = f"{PREFIX}/unified_dataset.jsonl"
    print(f"Uploading s3://{BUCKET}/{jsonl_key} ({len(jsonl_bytes):,} bytes)...")
    s3.put_object(Bucket=BUCKET, Key=jsonl_key, Body=jsonl_bytes)

    # Upload images zip
    zip_key = f"{PREFIX}/flickr30k-images.zip"
    zip_size = os.path.getsize(zip_path)
    print(f"Uploading s3://{BUCKET}/{zip_key} ({zip_size:,} bytes)...")
    with open(zip_path, "rb") as f:
        s3.put_object(Bucket=BUCKET, Key=zip_key, Body=f)

    # Extract and upload individual images
    print("Extracting and uploading individual images...")
    image_keys = []
    with zipfile.ZipFile(zip_path, "r") as zf:
        entries = [e for e in zf.infolist() if not e.is_dir()]
        total = len(entries)
        for i, entry in enumerate(entries, 1):
            filename = os.path.basename(entry.filename)
            if not filename:
                continue
            img_key = f"{PREFIX}/images/{filename}"
            with zf.open(entry) as img_file:
                s3.put_object(Bucket=BUCKET, Key=img_key, Body=img_file.read())
            image_keys.append(img_key)
            if i % 500 == 0 or i == total:
                print(f"  {i}/{total} images uploaded.")

    marker = {
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "record_count": len(records),
        "source": HF_REPO,
        "image_count": len(image_keys),
        "files": [json_key, jsonl_key, zip_key],
    }
    marker_key = f"{PREFIX}/ingest_done.json"
    print(f"Writing completion marker s3://{BUCKET}/{marker_key}...")
    s3.put_object(
        Bucket=BUCKET,
        Key=marker_key,
        Body=json.dumps(marker, indent=2).encode("utf-8"),
    )
    print("Done.")


if __name__ == "__main__":
    main()

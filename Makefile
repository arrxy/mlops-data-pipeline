build:
	docker compose build

db:
	docker compose up -d postgres

ingest:
	docker compose up -d postgres
	docker compose run --rm data-ingestor

generate:
	docker compose up -d postgres
	docker compose run --rm data-generator

features:
	docker compose run --rm feature-worker

compile:
	docker compose up -d postgres
	docker compose run --rm batch-compiler

index:
	docker compose up -d qdrant
	docker compose run --rm vector-indexer

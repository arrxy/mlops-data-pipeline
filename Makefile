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

worker:
	docker compose up -d postgres qdrant
	docker compose up online-feature-worker

init: build ingest features index
	@echo "Initial pipeline complete."

retrain: generate compile
	@echo "Retraining pipeline complete."

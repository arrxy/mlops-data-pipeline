build:
	docker compose build

ingest:
	docker compose run --rm data-ingestor

generate:
	docker compose run --rm data-generator

features:
	docker compose run --rm feature-worker

compile:
	docker compose run --rm batch-compiler

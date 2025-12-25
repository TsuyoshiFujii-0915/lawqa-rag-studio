CONFIG ?= config/config.yaml

# Run evaluation or serving with Docker

eval:
	docker compose run --rm app lawqa-rag-studio eval --config /app/$(CONFIG)

eval-recreate:
	docker compose run --rm app lawqa-rag-studio eval --config /app/$(CONFIG) --recreate-index

serve:
	@CONFIG=$(CONFIG) docker compose up -d qdrant app
	@CONFIG=$(CONFIG) docker compose logs -f app & \
	logs_pid=$$!; \
	trap 'docker compose stop app qdrant; kill $$logs_pid' INT TERM EXIT; \
	CONFIG=$(CONFIG) docker compose --profile frontend up --no-deps frontend

serve-recreate:
	@CONFIG=$(CONFIG) RECREATE_INDEX=1 docker compose up -d --force-recreate qdrant app
	@CONFIG=$(CONFIG) docker compose logs -f app & \
	logs_pid=$$!; \
	trap 'docker compose stop app qdrant; kill $$logs_pid' INT TERM EXIT; \
	CONFIG=$(CONFIG) docker compose --profile frontend up --no-deps frontend

# Run evaluation or serving with uv

eval-uv:
	uv run lawqa-rag-studio eval --config $(CONFIG)

eval-recreate-uv:
	uv run lawqa-rag-studio eval --config $(CONFIG) --recreate-index

serve-uv:
	@uv run lawqa-rag-studio serve --config $(CONFIG) & \
	backend_pid=$$!; \
	trap 'kill $$backend_pid' INT TERM EXIT; \
	cd frontend; \
	VITE_API_BASE=$${VITE_API_BASE:-http://localhost:8000} npm run dev

serve-recreate-uv:
	@uv run lawqa-rag-studio serve --config $(CONFIG) --recreate-index & \
	backend_pid=$$!; \
	trap 'kill $$backend_pid' INT TERM EXIT; \
	cd frontend; \
	VITE_API_BASE=$${VITE_API_BASE:-http://localhost:8000} npm run dev

# Other commands

up:
	docker compose up -d

down:
	docker compose down

build:
	docker compose build --no-cache app

gendocs:
	docker compose run --rm app python tools/generate_config_docs.py > docs/config_options.md

gendocs-uv:
	uv run python tools/generate_config_docs.py > docs/config_options.md

test:
	docker compose run --rm app python -m pytest

test-uv:
	uv run python -m pytest

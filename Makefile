# MAKEFILE
CONTAINER_NAME?=fastapi-interview

# The .PHONY rule is used to declare that 'test' and 'tests' are not files but rather commands.
# This prevents Make from checking for the existence of a file named 'test' or 'tests' and
# ensures that the recipes for these targets are always executed when requested.
.PHONY: test tests

build: ## Build image
	docker-compose -f docker-compose.dev.yaml build

deps: ## compile dependencies.
	@if [ "$(IGNORE_DOCKER)" != "1" ] && ! [ -f /.dockerenv ]; then \
		echo "Error: Dependencies must be compiled inside the Docker container."; \
		echo "Use 'make start' to start the container"; \
		exit 1; \
	fi
	python3 -m piptools compile --upgrade requirements/prod.in -o requirements.txt
	python3 -m piptools compile --upgrade requirements/dev.in -o requirements.dev.txt

start:
	@if [ -f /.dockerenv ]; then \
		echo "You are already inside the container :)"; \
	else \
		docker-compose -f docker-compose.dev.yaml up -d --build; \
		docker exec -it $(CONTAINER_NAME) bash; \
	fi

stop:
	docker-compose -f docker-compose.dev.yaml down

restart: stop start

app:
	uvicorn --host 0.0.0.0 --port 8081 src.main:application --reload

tests: ## compile dependencies.
	@if [ "$(IGNORE_DOCKER)" != "1" ] && ! [ -f /.dockerenv ]; then \
		echo "Error: Tests must be run inside the Docker container."; \
		echo "Use 'make start' to start the container"; \
		exit 1; \
	fi
	coverage run -m pytest --log-cli-level=INFO && coverage report

help:
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)
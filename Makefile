.PHONY: install lint typecheck test test-unit test-contracts smoke compile docker-build docker-up docker-down

install:
	pip install -r requirements.txt -r requirements-dev.txt

lint:
	ruff check src tests

typecheck:
	mypy src --ignore-missing-imports

compile:
	python -m compileall src tests

test: test-unit test-contracts

test-unit:
	pytest tests/unit -v

test-contracts:
	pytest tests/data_contracts -v

smoke:
	pytest tests/smoke -v

docker-build:
	docker build -t merchantmind:latest .

docker-up:
	docker compose up --build

docker-down:
	docker compose down

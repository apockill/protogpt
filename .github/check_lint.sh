#!/bin/bash
set -euxo pipefail

poetry run cruft check
poetry run mypy --ignore-missing-imports protogpt/ tests/
poetry run isort --check --diff protogpt/ tests/
poetry run black --check protogpt/ tests/
poetry run flake8 protogpt/ tests/ --darglint-ignore-regex '^test_.*'
poetry run bandit -r --severity medium high protogpt/ tests/
poetry run vulture --min-confidence 100 protogpt/ tests/
echo "Lint successful!"
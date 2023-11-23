#!/bin/bash
set -euxo pipefail

poetry run isort protogpt/ tests/
poetry run black protogpt/ tests/

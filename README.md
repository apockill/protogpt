# protogpt

A for-fun GPT prototype, heavily inspired by [Andrej Karpathy's tutorial](https://www.youtube.com/watch?v=kCc8FmEb1nY), with my own spin on the architecture.

The goal was to learn how to develop GPTs using clean code for easy to reproduce experimentation, but also
for easy deployment.

_________________

[![PyPI version](https://badge.fury.io/py/protogpt.svg)](http://badge.fury.io/py/protogpt)
[![Test Status](https://github.com/apockill/protogpt/workflows/Test/badge.svg?branch=main)](https://github.com/apockill/protogpt/actions?query=workflow%3ATest)
[![Lint Status](https://github.com/apockill/protogpt/workflows/Lint/badge.svg?branch=main)](https://github.com/apockill/protogpt/actions?query=workflow%3ALint)
[![codecov](https://codecov.io/gh/apockill/protogpt/branch/main/graph/badge.svg)](https://codecov.io/gh/apockill/protogpt)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://timothycrosley.github.io/isort/)
_________________

## Development

### Installing python dependencies
```shell
poetry install
```

### Running the code
```shell
poetry run train_gpt --datasets/tiny_shakespear.txt
```

### Running Tests
```shell
pytest .
```

### Formatting Code
```shell
bash .github/format.sh
```

### Linting
```shell
bash .github/check_lint.sh
```
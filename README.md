# Soccer Predictions

MCMC Soccer Predictions using StatsBomb Data.

## Development Setup

Install all project dependencies (including development tools like Black and Flake8) using Poetry:

```bash
poetry install
```

## Code Formatting and Linting

This project uses **Black** for code formatting and **Flake8** for code linting (configured in [pyproject.toml](file:///Users/paulotanaka/soccer-predictions/pyproject.toml) and [.flake8](file:///Users/paulotanaka/soccer-predictions/.flake8)).

### Check and Format Code
To automatically format the entire codebase with Black:
```bash
poetry run black .
```

### Run Linter
To check code quality and style compliance with Flake8:
```bash
poetry run flake8 .
```

## Running Unit Tests

The project includes a comprehensive unit test suite in the [test/](file:///Users/paulotanaka/soccer-predictions/test) directory covering all root and subdirectory modules.

To run all unit tests with verbose output:
```bash
PYTHONPATH=. poetry run python -m unittest discover -s test -p "*_test.py" -t . -v
```

> [!TIP]
> Setting `PYTHONPATH=.` and `-t .` (top-level directory) ensures Python properly discovers all top-level packages (such as [ml_model](file:///Users/paulotanaka/soccer-predictions/ml_model) and [utils.py](file:///Users/paulotanaka/soccer-predictions/utils.py)) during test discovery.

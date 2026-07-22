# Contributing to pyweight

Thank you for your interest in contributing to pyweight! This document covers
the basics to get you started.

## Development Setup

This project uses [uv](https://docs.astral.sh/uv/) for dependency management.

```sh
git clone https://github.com/AZX-PBC-OSS/pyweight.git
cd pyweight
uv sync --all-extras --group dev
```

## Common Tasks

The [Makefile](Makefile) provides convenient targets:

```sh
make lint        # Run ruff linter
make format      # Auto-format with ruff
make typecheck   # Run pyright type checker
make test        # Run pytest
make check       # lint + typecheck + test
```

## Commit Convention

This project uses [Conventional Commits](https://www.conventionalcommits.org/).
Each commit message should be structured as `<type>(<scope>): <description>`,
for example `feat(graph): add cycle detection` or `fix(cli): handle empty input`.

Common types: `feat`, `fix`, `docs`, `refactor`, `test`, `chore`, `ci`.

## Pull Requests

1. Fork the repository and create a branch from `main`.
2. Make your changes, keeping commits focused and conventionally formatted.
3. Run `make check` to ensure linting, types, and tests pass.
4. Open a pull request with a clear description of the change and motivation.

By contributing, you agree that your contributions will be licensed under the
project's MIT license.

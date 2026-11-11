# Contributing to OmniRAG

## Development setup

Use Python 3.11 and Node.js 22 or compatible versions.

```bash
python -m pip install -e ".[dev]"
cd frontend && npm ci && cd ..
```

Copy `.env.example` to `.env` and configure the provider and Qdrant settings required for the workflow being developed.

## Project boundaries

- Application runtime belongs in `backend/src/omnirag/`.
- Evaluation datasets, protocols, scorers, and CLI entry points belong in `eval/`.
- Operational and evaluation commands belong in `scripts/`.
- Tests belong in `tests/`, organized by unit, integration, evaluation, or diagnostic purpose.
- Deployment definitions belong in `docker/`.
- Technical documentation belongs in `docs/`.

Do not place evaluation-only dependencies in the production backend package.

## Validation

```bash
ruff check backend/src tests eval scripts
pytest
```

For frontend changes:

```bash
cd frontend
npm test
npm run build
```

For Docker changes, validate the relevant Compose files and production image build from the repository root.

## Pull requests

Use a focused branch and describe the problem, implementation, and validation performed. Preserve evaluation dataset versions and hashes. Do not commit secrets, private run artifacts, or unreviewed benchmark data. Update documentation and tests when behaviour or project structure changes.

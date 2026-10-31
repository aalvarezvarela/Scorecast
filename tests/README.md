# Tests

Run the complete suite from the repository root:

```sh
python -m pytest
python -m pytest tests/features/ -q
python -m pytest --collect-only -q
```

`pyproject.toml` supplies the import paths and limits discovery to `tests/`.
Every domain is a Python package; pytest uses its default import mode. CI runs
all domains on pull requests and pushes to `main`.

## Ownership

Choose the folder from the primary module being tested, not the provider name
or whether a test is a regression, integration, or parity comparison. For tests
covering several modules, use the public entry point whose behavior is asserted.
The initial assignment of every file is recorded in [MIGRATION.md](MIGRATION.md).

| Folder | Primary modules |
| --- | --- |
| `core/` | `nba_ou.config`, `nba_ou.utils` |
| `datasets/` | `nba_ou.create_training_data`, `nba_ou.data_processing.missing_data` |
| `features/` | Other `nba_ou.data_processing` modules |
| `ingestion/` | `nba_ou.fetch_data`, ingestion scripts, `lab.game_reports` |
| `training/` | `training_pipeline` outside reporting, `nba_ou.modeling` outside the registry, `lab.meta_learner`, campaign/preflight configuration |
| `prediction/` | `nba_ou.prediction` and prediction/retraining/scheduling entry points |
| `storage/` | `nba_ou.postgre_db`, `nba_ou.modeling.registry_*`, backup/registry archive scripts |
| `reporting/` | `training_pipeline.reporting` |

Examples: MoviOdds ingestion belongs in `ingestion/`; registry persistence belongs
in `storage/`; lineup feature assembly belongs in `features/`, even when the
feature is attached to an intermediate dataset. The prediction recorder belongs
in `prediction/`, including its PostgreSQL writer checks.

## Shared support

- Put reusable builders, fake clients and assertions in `tests/support/`.
  Import them through `tests.support`; do not import another `test_*.py` module.
- Keep reference implementations used for parity checks in `support/legacy/`.
- Use `tests.support.paths.REPO_ROOT` for files in the repository. Do not infer
  the root from a test file's depth.
- Keep fixtures local to their test module unless several modules need them.
  Use a domain's `conftest.py` for domain-wide fixtures and the root conftest
  only for suite-wide behavior. Return fresh mutable frames for each test.
- Keep test filenames and cases stable during organizational moves. Split a
  large module only in a subsequent behavior-focused change.

## Migration verification

Record collection before moving files and again afterward:

```sh
python -m pytest tests/ --collect-only -q > /tmp/tests-before.txt
python -m pytest tests/ --collect-only -q > /tmp/tests-after.txt
```

Compare sorted node IDs after replacing only the file path with its basename.
Keep the class, function and parameter ID, and preserve duplicate occurrences:

```python
from collections import Counter
from pathlib import Path


def collected(path):
    nodes = []
    for line in Path(path).read_text().splitlines():
        if line.startswith("tests/") and "::" in line:
            filename, case = line.split("::", 1)
            nodes.append(f"{Path(filename).name}::{case}")
    return Counter(nodes)


before = collected("/tmp/tests-before.txt")
after = collected("/tmp/tests-after.txt")
assert before, "The reference collection is empty"
assert before == after, (before - after, after - before)
```

Matching counts alone does not prove the same tests were preserved. Also run the
full suite after the move, since collection does not exercise runtime paths.

## External services and CI

The root `conftest.py` fails immediately if a test creates an unsimulated boto3
client or reaches a real HTTP, socket, or PostgreSQL connection. This failure
also escapes application `except Exception` fallbacks. Supply a fake through
`monkeypatch`; do not disable the guard for an entire domain. Tests specifically
about the local input cache disable its optional S3 tier; S3 behavior has its
own tests using `FakeS3Client`.

Five prediction-writer tests carry `@pytest.mark.postgres`. They can connect only
to the exact `NBA_OU_TEST_PG_DSN` value, using a local host or Unix socket. They
create and drop the `prediction` schema in that disposable database. Without a
DSN they are skipped locally; CI supplies PostgreSQL 16 and runs them with the
rest of the suite. The schema needs PostgreSQL 15+ (`NULLS NOT DISTINCT`), with
no Aiven-specific extensions.

```sh
NBA_OU_TEST_PG_DSN='postgresql://nba_tests:nba_tests@localhost:5432/nba_tests' \
  python -m pytest -m postgres -q
```

CI runs every test on pull requests and pushes to `main` with a `pytest-sugar`
progress bar (`--force-sugar` enables it without an interactive terminal), failure
details, and a final summary. GitHub Actions may render progress updates on
multiple lines. CI dumps every thread's stack if a test runs past 5 minutes and
uploads `test-results/pytest.xml` as `pytest-results` even if tests fail. The pip
dependency cache remains enabled. There is no reduced "fast" suite replacing the
full merge checks.

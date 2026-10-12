# Benchmarks

Performance benchmarks, run with [pytest-benchmark](https://pytest-benchmark.readthedocs.io).
They live outside `tests/`, so the regular test run doesn't pick them up.

```console
tox -e bench
# or, in an environment with `.[tests,bench]` installed:
pytest benchmarks -o addopts=
```

Both the in-memory and on-disk HEALPix mesh caches are disabled, so every round
measures a real build.

## Comparing against a baseline

Timings are only comparable on the same machine. Save a baseline, change the
code, then compare:

```console
pytest benchmarks -o addopts= --benchmark-save=base
# ... change code ...
pytest benchmarks -o addopts= --benchmark-compare=0001 --benchmark-compare-fail=min:30%
```

Results are stored in `.benchmarks/` (git-ignored).

## CI

On pull requests, the `benchmarks` job runs these benchmarks against the base
branch's `src/` and then against the PR, on the same runner. It fails when the
minimum time of any benchmark gets more than 30% slower.

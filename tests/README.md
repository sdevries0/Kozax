# Kozax Unit Tests

This folder provides lightweight regression tests for Kozax's public workflows.

## Scope

The tests are implementation-agnostic and focus on observable behavior:
- Population initialization
- Population evaluation
- Population evolution
- End-to-end `fit` execution
- Mutation output invariants
- Expression rendering

## Run

From the repository root:

```bash
python -m pytest -q tests
```

Or run a single file:

```bash
python -m pytest -q tests/test_gp_workflows.py
```

## Notes

- These tests are intentionally small and fast.
- They are based on usage patterns from example scripts, but use toy data to keep runtime low.

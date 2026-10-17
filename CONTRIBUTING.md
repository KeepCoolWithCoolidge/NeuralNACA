# Contributing

Thank you for helping! Bug reports, questions and pull requests are welcome.

## Development setup

```bash
git clone https://github.com/KeepCoolWithCoolidge/NeuralNACA.git && cd NeuralNACA
python -m pip install -e ".[dev]"
bash scripts/install_xfoil_python.sh          # optional: needed for XFOIL tests and new data
```

## Checks (the same ones CI runs)

```bash
ruff check src tests scripts examples         # lint
mypy                                          # types
pytest -m "not slow"                          # fast suite (< 1 min, no XFOIL needed)
pytest                                        # everything available on your machine (XFOIL tests skip without XFOIL)
neuralnaca reproduce --level 0                # the shipped results still verify
mkdocs build --strict                         # documentation
```

Markers: `slow` (trains networks), `xfoil` (runs XFOIL), `neuralfoil` (needs the neuralfoil package), `benchmark`
(needs the shipped `data/` and `benchmarks/` files). Property-based tests are deterministic by default; set
`HYPOTHESIS_PROFILE=thorough` for a longer randomised search.

## Rules for changes

* **Anything that changes predictions** (features, architecture, panel discretisation, training recipe) must bump
  `FEATURES_VERSION` in `neuralnaca/model.py` if old weights become incompatible, retrain the shipped model, re-run the
  benchmark (`neuralnaca bench ...`, see docs/reference/cli.md), regenerate tables and figures
  (`neuralnaca report && neuralnaca figures`), update `benchmarks/MANIFEST.json` (`python scripts/make_manifest.py`)
  and the README headline, and be described in `CHANGELOG.md`. The integrity tests enforce consistency.
* Numbers in the documentation are never typed by hand: they come from `docs/results/tables.md`, which is generated.
* Keep negative results. If an idea does not help, document it rather than dropping it silently.
* New behaviour needs tests; bug fixes need a regression test.

## Reporting a discrepancy with the published results

Please run `neuralnaca reproduce --level 0 --json report.json` and `neuralnaca info`, and attach both files to the
issue.

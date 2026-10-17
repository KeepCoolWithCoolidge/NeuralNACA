# Python API

The package is organised in layers; most users only need the first.

| module | purpose |
|---|---|
| `neuralnaca` | `get_aero`, `NeuralNACA`, `parse_naca`, `naca4_coordinates`, `DEFAULT_DOMAIN`, warnings |
| `neuralnaca.domain` | the validated domain and the sampling designs |
| `neuralnaca.geometry` | NACA 4-digit geometry (NumPy and differentiable PyTorch) |
| `neuralnaca.panel` | the differentiable Hess-Smith panel method |
| `neuralnaca.model` | features, network, symmetry check, weights files |
| `neuralnaca.training` | the training recipe |
| `neuralnaca.support` | training-support estimate |
| `neuralnaca.xfoil` | XFOIL configuration and batch driver |
| `neuralnaca.data` | XFOIL data sets, generation, integrity |
| `neuralnaca.evaluation` | metrics and paired statistics |
| `neuralnaca.baselines` | NeuralFoil wrapper |
| `neuralnaca.benchmark` | the experiments behind every published number |
| `neuralnaca.report`, `neuralnaca.plotting` | generated tables and figures |

## Predictions

::: neuralnaca.api.NeuralNACA

::: neuralnaca.api.get_aero

::: neuralnaca.domain.Domain

::: neuralnaca.domain.DomainWarning

::: neuralnaca.domain.UnsupportedWarning

## Geometry

::: neuralnaca.geometry

## Panel method

::: neuralnaca.panel

## Model

::: neuralnaca.model

## Training

::: neuralnaca.training

## Training support

::: neuralnaca.support

## XFOIL

::: neuralnaca.xfoil

## Data

::: neuralnaca.data

::: neuralnaca.domain.sample_points

## Evaluation

::: neuralnaca.evaluation

::: neuralnaca.baselines

## Benchmark

::: neuralnaca.benchmark

# Critical brain: exploratory spiking-network models

A collection of early network simulations and avalanche plots. The scripts are
prototypes: their names and plots do not establish that a network is critical.

## Contents

- `dcm_izhik.py`: single-population and multi-population Izhikevich-style demos,
  raster plots, and descriptive avalanche analysis. Importing it does not run
  either demo.
- `avalanche_analysis.py`: dependency-free analysis of recorded spike bins and
  an empirical complementary cumulative distribution.
- `Dynamics.py`: explicit launcher for the separate historical grid experiment.
  Importing it does not load the scientific runtime or run simulations.
- `legacy/dynamics_demo.py`: preserved original grid simulation and plots.
- [Grid-model contract review](docs/dynamics-contract.md): indexing, spike timing,
  weights, and avalanche boundaries that need a model decision before correction.
- `critical_brain_readme.pdf` and the images: original project notes, preserved.

## Set up and run

Use a project environment with Python 3.11 or newer:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python dcm_izhik.py
```

The last command explicitly runs both demonstrations and opens plots. To choose
one, import `run_single_population_demo` or `run_multi_agent_demo` and call it.
`python Dynamics.py` runs the separate historical experiment. These are not
smoke-test commands: they perform simulations. Dependency ranges are supplied;
an exact lock and reproducible scientific seeds are still future work.

## Fast checks without scientific simulations

```sh
python -m unittest discover -s tests -v
```

Tests use hand-written spike bins and stub populations. They check that a
recording contains one independent frame per timestep, that avalanches are not
counted repeatedly or dropped at the end, and that the CCDF has a positive tail.
No long demo or scientific training runs in these checks. GitHub Actions runs
the same suite with the declared dependencies.

`all_firings[t][agent]` is the list/array of neuron indices for that timestep,
not the agent's entire accumulated history. Existing callers relying on the old
aliased history shape must adopt this explicit recording format.

## Interpretation and remaining work

- `fit_power_law` performs a descriptive log-log regression on an empirical
  survival curve; it is not a power-law hypothesis test or evidence of
  criticality. Time-bin choices and model comparison must be specified before
  scientific interpretation.
- A final active segment is included in the observed sizes/durations. It may
  be right-censored and must not automatically be treated as a complete event.
- `MultiAgentNetwork.interactions` is initialized but not applied by `step`.
  The current populations are not a tested coupled multi-agent model.
- In `legacy/dynamics_demo.py`, the sampled excitatory indices and contiguous index slices
  are inconsistent, and coupling reads membrane thresholds after immediate
  resets. Its connectivity/spike-update semantics need an explicit model
  decision and focused tests before interpreting new results.
- Random seeds, simulation configuration, and saved-output provenance are not
  standardized yet. Existing notes and images were not regenerated here.

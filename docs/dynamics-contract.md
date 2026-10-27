# Historical grid model: current behavior and open decisions

`Dynamics.py` is an explicit launcher. Importing it only exposes `run_demo()`;
`python Dynamics.py` executes the preserved script in
[`legacy/dynamics_demo.py`](../legacy/dynamics_demo.py). The legacy file is a
demonstration script, not an importable model API. Its source has been retained
without changing its numerical implementation or running its experiments.

Before correcting that model, resolve these concrete inconsistencies:

| Area | Current implementation | Contract needed for a correction |
|---|---|---|
| Neuron identity | Excitatory indices are sampled randomly, but synaptic inputs and inter-spike intervals also use contiguous `[:n_excitatory]` slices. | Use a single explicit identity map consistently for connectivity, updates, and statistics. |
| Spike delivery | A neuron resets its voltage immediately; coupling reads `V > threshold`, rather than a recorded spike mask. Updates run sequentially in neuron-index order. | Specify whether spikes from the previous or current step drive inputs, the delivery delay, and synchronous versus sequential updates. |
| Connection weights | Both `W_IE` and `W_EE` are applied to the same excitatory slice for every target, and both `W_EI` and `W_II` to the inhibitory slice. | Define source/target meanings of each weight and select the intended pair from neuron identities. |
| Avalanche boundaries | `simulate()` starts another size entry after an active bin instead of accumulating a contiguous active run. | Choose per-agent versus pooled activity, bin width, and handling of censored boundary segments before comparing distributions. |
| Regime labels | Three chosen currents are labelled subcritical/critical/supercritical. | Treat these as historical plot labels until a separate protocol and measurements justify the classification. |

The tested `avalanche_analysis.py` helpers belong to the explicit spike-frame
format used by `dcm_izhik.py`. They have not silently replaced the historical
grid model's analysis. The grid experiment and stored images must not be treated
as newly reproduced results because the launcher was refactored.

The separate [opt-in grid transition contract](grid-step-contract.md) now defines
one deterministic transition with fully explicit inputs and a fixed arithmetic
fixture. Its synchronous, previous-bin delivery semantics are new engineering
choices, not a resolution of what the historical author intended. It does not
change `legacy/dynamics_demo.py` or launch an experiment. Random sampling remains
outside that API and must receive an explicit RNG if introduced later.

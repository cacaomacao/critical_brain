# Opt-in grid transition contract

This is a new, deterministic engineering contract for `grid_step.step`. It is
not a reconstruction, correction, or reproduction of `legacy/dynamics_demo.py`.
Neither the historical launcher nor `dcm_izhik.py` calls it. The choices below
make one transition reviewable; they do not identify a scientific regime or
justify parameters for experiments. There are no parameter defaults or random
draws, and no connectivity, initial-state, or E/I-map generator.

## Complete inputs and identity

`GridConfig` requires every field below. Numeric values must be finite; time
constants and `dt_ms` must be strictly positive. Input sequences are copied to
tuples, so later caller changes cannot alter a configuration or state.

| Field | Meaning |
|---|---|
| `grid_shape` | Two positive integers `(rows, columns)`; `N = rows * columns`. Index `i` is cell `(i // columns, i % columns)`. Geometry only labels cells: no distance rule, boundary wrapping, or implicit edges. |
| `neuron_types` | Exactly `N` entries, each `"E"` or `"I"`, in cell-index order. This single map selects membrane time constants, resets, and source/target weights. Neither type is required to occupy a contiguous slice or to be present. |
| `connectivity` | An `N x N` Boolean matrix. Entry `[source][target]` enables one directed edge. Self-edges are allowed only when explicitly `True`; no edges are added or inferred. |
| `weights` | `SynapticWeights` with four required, signed numbers: `excitatory_to_excitatory`, `excitatory_to_inhibitory`, `inhibitory_to_excitatory`, `inhibitory_to_inhibitory`. Names always mean **source to target**. Signs are caller supplied, never inferred, negated, or constrained from the type label. |
| `dt_ms` | Duration of one transition and one returned spike bin. |
| `tau_excitatory_ms`, `tau_inhibitory_ms` | Membrane decay time constants, selected by the target's identity. |
| `tau_synaptic_ms` | Shared synaptic-current decay time constant. |
| `threshold` | A neuron spikes only if its updated, pre-reset voltage is **strictly greater** than this value. Equality does not spike. |
| `reset_excitatory`, `reset_inhibitory` | Post-spike voltages, selected by the spiking neuron's identity. No implicit refractory period or second threshold check after reset. |

`GridState` contains exactly `N` finite `voltage` values, `N` finite
`synaptic_current` values, and an `N`-entry Boolean `spikes` mask. At entry to
transition `k`, these are `V_k`, `I_k`, and the spikes emitted in the **previous**
bin, `S_(k-1)`. The caller explicitly supplies an initial mask, normally all
`False` for a recording with no preceding activity; initialization is not part
of this function. `external_current` is a required, finite `N`-entry vector held
constant during the transition. There is no scalar broadcast or implicit drive.

## One synchronous transition

For every target `j`, select `W(type[source], type[j])` from the four named
weights and compute from the unchanged input state:

```text
drive_j = sum(connectivity[source][j] * W(type[source], type[j])
              * S_(k-1)[source] for source in range(N))
I_(k+1)[j] = I_k[j] + dt_ms * (-I_k[j] / tau_synaptic_ms + drive_j)
trial_j = V_k[j] + dt_ms * (-V_k[j] / tau(type[j])
                           + I_(k+1)[j] + external_current[j])
S_k[j] = trial_j > threshold
V_(k+1)[j] = reset(type[j]) if S_k[j] else trial_j
```

`step(config, state, external_current)` returns a new `GridState` containing
`V_(k+1)`, `I_(k+1)`, and **current** mask `S_k`. It never mutates the inputs.
Passing this returned state to the next call delivers `S_k` exactly one bin
later. Current spikes never feed the same call, including through self-edges.
Initial voltage alone cannot stand in for a recorded previous spike.

This contract uses one Euler update for current followed by one voltage update
with that new current. Weights contribute a current derivative per active
source bin, so their contribution is multiplied by `dt_ms`; they are not
instantaneous current jumps. With voltage unit `U`, current and external drive
have unit `U/ms`, weights have unit `U/ms^2`, and all time values are in ms.
There is no substepping, noise, saturation, normalization by population size,
or stability guarantee. Nonfinite arithmetic results are rejected rather than
returned. Large or unsuitable finite parameters still require scientific
review. Random sampling is outside this API; a future sampling API must accept
an explicit RNG and supply its realized inputs to this deterministic step.

## Fixed one-transition fixture

These numbers are an arithmetic fixture, not proposed scientific parameters.
They exercise interleaved identities and all four directed type combinations.

```text
grid_shape = (2, 2)
neuron_types = ("I", "E", "I", "E")
connectivity = ((False, True,  True,  False),
                (True,  False, False, True),
                (False, False, False, True),
                (False, False, True,  False))
weights = (E_to_E=1, E_to_I=4, I_to_E=-2, I_to_I=-3)
dt_ms = 0.5
tau_excitatory_ms = 2; tau_inhibitory_ms = 4; tau_synaptic_ms = 2
threshold = 1; reset_excitatory = -2; reset_inhibitory = -3
V_k = (0, 0.5, -0.5, 1)
I_k = (0, 2, -2, 0)
S_(k-1) = (True, True, False, False)
external_current = (0, 1, 0, 0)

drive = (4, -2, -3, 1)
I_(k+1) = (2, 0.5, -3, 0.5)
trial = (1, 1.125, -1.9375, 1)
S_k = (False, True, False, False)
V_(k+1) = (1, -2, -1.9375, 1)
```

## Recording and avalanche boundary convention

One returned mask is one bin of width `dt_ms`; record masks in call order,
without replaying the input mask as a newly observed bin. Within this contract,
one grid is one pooled population. An active bin contains at least one spike;
an avalanche is a maximal contiguous run of active bins. Size is the sum of
spike counts in the run; duration is its bin count (or that count times
`dt_ms`). A silent bin closes the preceding run. A run touching the recording's
first or last bin is potentially left- or right-censored, respectively, unless
silence outside the recording is independently known. It must retain that
boundary flag before analysis as a complete event. Multiple grids, other bin
widths, and per-type events need another explicit analysis contract.

The step returns masks only; it does not estimate avalanches or classify
criticality. The existing `avalanche_analysis.py` interface for `dcm_izhik.py`
is not silently reused or changed by this contract.

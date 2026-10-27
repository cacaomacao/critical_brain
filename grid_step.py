"""Opt-in deterministic grid transition; see docs/grid-step-contract.md.

This module has no demos, random draws, scientific-runtime imports, or defaults
for model parameters. It does not reproduce the historical grid experiment.
"""

from dataclasses import dataclass
from math import fsum, isfinite
from numbers import Integral, Real


def _as_tuple(name, values):
    try:
        return tuple(values)
    except TypeError as error:
        raise ValueError(f"{name} must be a sequence") from error


def _finite_number(name, value):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    try:
        number = float(value)
    except OverflowError as error:
        raise ValueError(f"{name} must be finite") from error
    if not isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _finite_vector(name, values):
    return tuple(_finite_number(name, value)
                 for value in _as_tuple(name, values))


def _boolean_vector(name, values):
    result = _as_tuple(name, values)
    if any(type(value) is not bool for value in result):
        raise ValueError(f"{name} must contain Boolean values")
    return result


@dataclass(frozen=True)
class SynapticWeights:
    """Signed current-derivative contributions; names mean source to target."""

    excitatory_to_excitatory: float
    excitatory_to_inhibitory: float
    inhibitory_to_excitatory: float
    inhibitory_to_inhibitory: float

    def __post_init__(self):
        for name in self.__dataclass_fields__:
            object.__setattr__(self, name, _finite_number(name, getattr(self, name)))


@dataclass(frozen=True)
class GridConfig:
    """Complete, immutable configuration, with no inferred edges or identities."""

    grid_shape: tuple[int, int]
    neuron_types: tuple[str, ...]
    connectivity: tuple[tuple[bool, ...], ...]
    weights: SynapticWeights
    dt_ms: float
    tau_excitatory_ms: float
    tau_inhibitory_ms: float
    tau_synaptic_ms: float
    threshold: float
    reset_excitatory: float
    reset_inhibitory: float

    def __post_init__(self):
        shape = _as_tuple("grid_shape", self.grid_shape)
        if len(shape) != 2 or any(isinstance(value, bool)
                                 or not isinstance(value, Integral) or value <= 0
                                 for value in shape):
            raise ValueError("grid_shape must contain two positive integers")
        shape = tuple(int(value) for value in shape)
        count = shape[0] * shape[1]
        identities = _as_tuple("neuron_types", self.neuron_types)
        if len(identities) != count or any(type(value) is not str
                                          or value not in ("E", "I")
                                          for value in identities):
            raise ValueError("neuron_types must contain one E/I identity per cell")
        connectivity = tuple(_boolean_vector("connectivity", row)
                             for row in _as_tuple("connectivity", self.connectivity))
        if len(connectivity) != count or any(len(row) != count for row in connectivity):
            raise ValueError("connectivity must be an N x N source/target matrix")
        if not isinstance(self.weights, SynapticWeights):
            raise ValueError("weights must be SynapticWeights")
        object.__setattr__(self, "grid_shape", shape)
        object.__setattr__(self, "neuron_types", identities)
        object.__setattr__(self, "connectivity", connectivity)
        for name in ("dt_ms", "tau_excitatory_ms", "tau_inhibitory_ms",
                     "tau_synaptic_ms", "threshold", "reset_excitatory",
                     "reset_inhibitory"):
            value = _finite_number(name, getattr(self, name))
            if name.endswith("_ms") and value <= 0:
                raise ValueError(f"{name} must be strictly positive")
            object.__setattr__(self, name, value)

    @property
    def n_neurons(self):
        return self.grid_shape[0] * self.grid_shape[1]


@dataclass(frozen=True)
class GridState:
    """Boundary values and last bin's spike mask, copied to immutable tuples."""

    voltage: tuple[float, ...]
    synaptic_current: tuple[float, ...]
    spikes: tuple[bool, ...]

    def __post_init__(self):
        voltage = _finite_vector("voltage", self.voltage)
        current = _finite_vector("synaptic_current", self.synaptic_current)
        spikes = _boolean_vector("spikes", self.spikes)
        if not voltage or len(voltage) != len(current) or len(voltage) != len(spikes):
            raise ValueError("state vectors must have the same nonzero length")
        object.__setattr__(self, "voltage", voltage)
        object.__setattr__(self, "synaptic_current", current)
        object.__setattr__(self, "spikes", spikes)


def step(config: GridConfig, state: GridState,
         external_current: tuple[float, ...]) -> GridState:
    """Advance one bin synchronously, delivering only ``state.spikes``.

    The returned state's mask contains this bin's spikes and is delivered by
    the next call. Weights enter the current derivative and are scaled by dt.
    No input is mutated; no randomness or historical implementation is used.
    """
    if not isinstance(config, GridConfig) or not isinstance(state, GridState):
        raise ValueError("step requires GridConfig and GridState")
    external = _finite_vector("external_current", external_current)
    count = config.n_neurons
    if len(state.voltage) != count or len(external) != count:
        raise ValueError("state and external_current must have one entry per cell")

    weights = {
        ("E", "E"): config.weights.excitatory_to_excitatory,
        ("E", "I"): config.weights.excitatory_to_inhibitory,
        ("I", "E"): config.weights.inhibitory_to_excitatory,
        ("I", "I"): config.weights.inhibitory_to_inhibitory,
    }
    try:
        drive = tuple(fsum(weights[config.neuron_types[source], config.neuron_types[target]]
                           for source in range(count)
                           if state.spikes[source] and config.connectivity[source][target])
                      for target in range(count))
        current = _finite_vector("updated synaptic_current", (
            value + config.dt_ms * (-value / config.tau_synaptic_ms + drive[target])
            for target, value in enumerate(state.synaptic_current)))
        trial = _finite_vector("pre-reset voltage", (
            value + config.dt_ms * (
                -value / (config.tau_excitatory_ms if config.neuron_types[target] == "E"
                          else config.tau_inhibitory_ms)
                + current[target] + external[target])
            for target, value in enumerate(state.voltage)))
    except OverflowError as error:
        raise ValueError("transition arithmetic must remain finite") from error

    spikes = tuple(value > config.threshold for value in trial)
    voltage = tuple((config.reset_excitatory if identity == "E" else config.reset_inhibitory)
                    if spike else value
                    for value, identity, spike in zip(trial, config.neuron_types, spikes))
    return GridState(voltage, current, spikes)

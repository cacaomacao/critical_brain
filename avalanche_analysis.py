"""Small dependency-free helpers for one-frame-per-timestep spike recordings."""
from collections import Counter
import math


def analyze_avalanches(all_firings, num_agents):
    """Return spike counts and bin durations for contiguous active bins per agent.

    ``all_firings[t][agent]`` contains only the neuron indices that fired in that
    timestep. A silent bin ends an avalanche; final activity is included as an
    observed segment, which is right-censored if recording stops mid-avalanche.
    The bin width is supplied by the recording, not inferred here.
    """
    if num_agents < 1:
        raise ValueError('num_agents must be positive.')
    frames = list(all_firings)
    if any(len(frame) != num_agents for frame in frames):
        raise ValueError('Each timestep must contain exactly num_agents recordings.')
    sizes, durations = [], []
    for agent in range(num_agents):
        size = duration = 0
        for frame in frames:
            count = len(frame[agent])
            if count:
                size += count
                duration += 1
            elif duration:
                sizes.append(size)
                durations.append(duration)
                size = duration = 0
        if duration:
            sizes.append(size)
            durations.append(duration)
    return sizes, durations


def empirical_ccdf(data):
    """Return unique positive values and P(X >= value), including tied samples."""
    values = [float(value) for value in data]
    if any(not math.isfinite(value) for value in values):
        raise ValueError('Samples must be finite.')
    counts = Counter(value for value in values if value > 0)
    if not counts:
        return [], []
    support = sorted(counts)
    remaining = sum(counts.values())
    total = remaining
    survival = []
    for value in support:
        survival.append(remaining / total)
        remaining -= counts[value]
    return support, survival

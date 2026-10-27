"""Exploratory spiking models and descriptive avalanche plots.

Importing this module defines models only. Run the demonstrations explicitly.
"""
import numpy as np
import matplotlib.pyplot as plt

from avalanche_analysis import analyze_avalanches, empirical_ccdf


def run_single_population_demo():
    # Parameters
    Ne = 800  # Number of excitatory neurons
    Ni = 200  # Number of inhibitory neurons
    re = np.random.rand(Ne)
    ri = np.random.rand(Ni)
    a = np.concatenate((0.02 * np.ones(Ne), 0.02 + 0.08 * ri))
    b = np.concatenate((0.2 * np.ones(Ne), 0.25 - 0.05 * ri))
    c = np.concatenate((-65 + 15 * re**2, -65 * np.ones(Ni)))
    d = np.concatenate((8 - 6 * re**2, 2 * np.ones(Ni)))
    S = np.hstack((0.5 * np.random.rand(Ne + Ni, Ne), -np.random.rand(Ne + Ni, Ni)))
    v = -65 * np.ones(Ne + Ni)  # Initial values of v
    u = b * v  # Initial values of u
    firings = []  # Spike timings

    # Simulation parameters
    time_steps = 1000  # Number of simulation steps

    # Simulation loop
    for t in range(time_steps):
        I = np.concatenate((5 * np.random.randn(Ne), 2 * np.random.randn(Ni)))  # Thalamic input
        fired = np.where(v >= 30)[0]  # Indices of spikes
        firings.extend([(t, neuron) for neuron in fired])
        v[fired] = c[fired]
        u[fired] += d[fired]
        I += np.sum(S[:, fired], axis=1)
        v = v + 0.5 * (0.04 * v**2 + 5 * v + 140 - u + I)  # Step 0.5 ms
        v = v + 0.5 * (0.04 * v**2 + 5 * v + 140 - u + I)  # for numerical stability
        u = u + a * (b * v - u)

    # Convert firings to a numpy array for easier handling
    firings = np.array(firings)
    # Plot the spikes
    plt.figure(figsize=(12, 8))
    plt.scatter(firings[:, 0], firings[:, 1], s=1)
    plt.title('Raster plot of neuronal spikes')
    plt.xlabel('Time (ms)')
    plt.ylabel('Neuron index')
    plt.show()


class NeuronalPopulation:
    def __init__(self, N=100, excitatory_ratio=0.8):
        self.N = N
        self.exc_neurons = int(N * excitatory_ratio)
        self.inh_neurons = N - self.exc_neurons
        self.v = np.random.randn(N) * -65  # Membrane potential
        self.u = np.zeros(N)  # Recovery variable
        self.a = np.concatenate((0.02 * np.ones(self.exc_neurons), 0.02 + 0.08 * np.random.rand(self.inh_neurons)))
        self.b = np.concatenate((0.2 * np.ones(self.exc_neurons), 0.25 - 0.05 * np.random.rand(self.inh_neurons)))
        self.c = np.concatenate((-65 + 15 * np.random.rand(self.exc_neurons)**2, -65 * np.ones(self.inh_neurons)))
        self.d = np.concatenate((8 - 6 * np.random.rand(self.exc_neurons)**2, 2 * np.ones(self.inh_neurons)))
        self.S = np.random.rand(N, N) - 0.5  # Synaptic weights
        self.firings = []

    def step(self, I):
        fired = np.where(self.v >= 30)[0]
        self.firings.append(fired)
        self.v[fired] = self.c[fired]
        self.u[fired] += self.d[fired]
        I += np.sum(self.S[:, fired], axis=1)
        self.v += 0.5 * (0.04 * self.v**2 + 5 * self.v + 140 - self.u + I)
        self.v += 0.5 * (0.04 * self.v**2 + 5 * self.v + 140 - self.u + I)
        self.u += self.a * (self.b * self.v - self.u)
        return self.v, self.u, self.firings

class MultiAgentNetwork:
    def __init__(self, num_agents=5, N=100, excitatory_ratio=0.8):
        self.agents = [NeuronalPopulation(N, excitatory_ratio) for _ in range(num_agents)]
        self.interactions = np.random.rand(num_agents, num_agents) - 0.5

    def step(self):
        frame = []
        for agent in self.agents:
            external_input = np.random.randn(agent.N) * 5
            _, _, history = agent.step(external_input)
            frame.append(history[-1].copy())
        return frame

    def simulate(self, steps=1000):
        # One independent frame per timestep, not aliases of cumulative history.
        return [self.step() for _ in range(steps)]


def plot_firings(all_firings, neurons_per_agent=100):
    plt.figure(figsize=(12, 8))
    for time_step, frame in enumerate(all_firings):
        for agent_idx, neurons in enumerate(frame):
            plt.scatter([time_step] * len(neurons),
                        np.asarray(neurons) + agent_idx * neurons_per_agent, s=1)
    plt.title('Raster plot of neuronal spikes across agents')
    plt.xlabel('Time (ms)')
    plt.ylabel('Neuron index (with agent offset)')
    plt.show()


def plot_distribution(data, title, xlabel, ylabel):
    plt.figure()
    plt.hist(data, bins=30, density=True)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()


def fit_power_law(data, show=True):
    """Return a descriptive log-log regression, not a power-law hypothesis test."""
    values, survival = empirical_ccdf(data)
    if len(values) < 2:
        raise ValueError('At least two distinct positive values are needed for a log-log regression.')
    from scipy.stats import linregress

    values = np.asarray(values)
    survival = np.asarray(survival)
    fit = linregress(np.log(values), np.log(survival))
    plt.figure()
    plt.loglog(values, survival, marker='.', linestyle='none')
    plt.plot(values, np.exp(fit.intercept + fit.slope * np.log(values)),
             label=f'Slope = {fit.slope:.2f}')
    plt.title('Descriptive log-log CCDF regression')
    plt.xlabel('Data')
    plt.ylabel('P(X >= x)')
    plt.legend()
    if show:
        plt.show()
    return fit.slope, fit.intercept, fit.rvalue, fit.pvalue, fit.stderr


def run_multi_agent_demo():
    network = MultiAgentNetwork(num_agents=5)
    all_firings = network.simulate(steps=1000)
    plot_firings(all_firings, neurons_per_agent=network.agents[0].N)
    sizes, durations = analyze_avalanches(all_firings, num_agents=5)
    plot_distribution(sizes, 'Avalanche Size Distribution', 'Size', 'Frequency')
    plot_distribution(durations, 'Avalanche Duration Distribution', 'Duration', 'Frequency')
    if len(set(sizes)) >= 2:
        slope, intercept, r_value, p_value, std_err = fit_power_law(sizes)
        print(f'Descriptive fit: slope={slope:.2f}, intercept={intercept:.2f}, '
              f'r_value={r_value:.2f}, p_value={p_value:.2e}, std_err={std_err:.2f}')
    else:
        print('Not enough distinct avalanche sizes for a log-log regression.')


if __name__ == '__main__':
    run_single_population_demo()
    run_multi_agent_demo()

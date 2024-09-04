import numpy as np
import matplotlib.pyplot as plt

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

#DCM
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

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
        self.interactions = np.random.rand(num_agents, num_agents) - 0.5  # Interaction weights between agents

    def step(self):
        for agent in self.agents:
            external_input = np.random.randn(agent.N) * 5  # Random thalamic input
            agent.step(external_input)

    def simulate(self, steps=1000):
        all_firings = []
        for _ in range(steps):
            self.step()
            all_firings.append([agent.firings for agent in self.agents])
        return all_firings

#simulation
network = MultiAgentNetwork(num_agents=5)
all_firings = network.simulate(steps=1000)


#plotting
def plot_firings(all_firings):
    plt.figure(figsize=(12, 8))
    for agent_idx, agent_firings in enumerate(all_firings[-1]):
        for time_step, neurons in enumerate(agent_firings):
            plt.scatter([time_step] * len(neurons), neurons + agent_idx * 100, s=1)  # Offset by agent index
    plt.title('Raster plot of neuronal spikes across agents')
    plt.xlabel('Time (ms)')
    plt.ylabel('Neuron index (with agent offset)')
    plt.show()

plot_firings(all_firings)

#criticality analysis
def analyze_avalanches(all_firings, num_agents):
    avalanche_sizes = []
    avalanche_durations = []

    for agent_idx in range(num_agents):
        agent_firings = [firing for step_firings in all_firings for firing in step_firings[agent_idx]]
        current_avalanche_size = 0
        current_avalanche_duration = 0

        for step in agent_firings:
            if len(step) > 0:
                current_avalanche_size += len(step)
                current_avalanche_duration += 1
            elif current_avalanche_size > 0:
                avalanche_sizes.append(current_avalanche_size)
                avalanche_durations.append(current_avalanche_duration)
                current_avalanche_size = 0
                current_avalanche_duration = 0

    return avalanche_sizes, avalanche_durations

def plot_distribution(data, title, xlabel, ylabel):
    plt.figure()
    plt.hist(data, bins=30, density=True)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

def fit_power_law(data):
    data = np.array(data)
    data = data[data > 0]  # Remove zeros
    data = np.sort(data)
    cdf = np.arange(1, len(data) + 1) / len(data)

    plt.figure()
    plt.loglog(data, 1 - cdf, marker='.', linestyle='none')
    plt.title('Power-law fit')
    plt.xlabel('Data')
    plt.ylabel('CCDF')

    slope, intercept, r_value, p_value, std_err = linregress(np.log(data), np.log(1 - cdf))
    plt.plot(data, np.exp(intercept + slope * np.log(data)), label=f'Slope = {slope:.2f}')
    plt.legend()
    plt.show()

    return slope, intercept, r_value, p_value, std_err

# Analyze avalanches
avalanche_sizes, avalanche_durations = analyze_avalanches(all_firings, num_agents=5)

# Plot avalanche size distribution
plot_distribution(avalanche_sizes, 'Avalanche Size Distribution', 'Size', 'Frequency')

# Plot avalanche duration distribution
plot_distribution(avalanche_durations, 'Avalanche Duration Distribution', 'Duration', 'Frequency')

# Fit power-law to avalanche size distribution
slope, intercept, r_value, p_value, std_err = fit_power_law(avalanche_sizes)
print(f'Power-law fit: slope={slope:.2f}, intercept={intercept:.2f}, r_value={r_value:.2f}, p_value={p_value:.2e}, std_err={std_err:.2f}')

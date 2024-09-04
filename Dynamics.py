import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy
from scipy.optimize import curve_fit
import matplotlib.animation as animation

# Network parameters
n_neurons = 100
n_excitatory = int(0.75 * n_neurons)
excitatory_neurons = np.random.choice(range(n_neurons), n_excitatory, replace=False) #list of the indices of the exc neurons
n_inhibitory = n_neurons - n_excitatory
inhibitory_neurons = list(set(range(n_neurons)) - set(excitatory_neurons)) #list of the indices of the inh neurons
grid_size = 10
connectivity_range = 7
exc_connectivity_prob = 0.2
inh_connectivity_prob = 0.3
simulation_time = 2000  # in ms
dt = 1  # in ms

# Neuron parameters
tau_exc = 6  # in ms
tau_inh = 12  # in ms
tau_I = 9  # in ms
W_IE = 0.2
W_EE = 0.05
W_EI = -2
W_II = -2
I0 = 0.2  # Added constant input current
P0_exc = 0.000001  # in 1/ms
P0_inh = 0  # in 1/ms
Pr_exc = -2  # in 1/ms
Pr_inh = -20  # in 1/ms
threshold = 1  # Spiking threshold

# Connectivity matrix
def distance(i, j):
    #this function accepts two neuron indexes (each one in the grid have a coardinate x,y)
    #and calculates the distance between the two neurons.
    x1, y1 = i // grid_size, i % grid_size
    x2, y2 = j // grid_size, j % grid_size
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

def connection_probability(i, j):
    d = distance(i, j)
    if d <= connectivity_range:
        if i in excitatory_neurons and j in excitatory_neurons:
            return exc_connectivity_prob * np.exp(-d)
        if i not in excitatory_neurons and j not in excitatory_neurons:
            return inh_connectivity_prob * np.exp(-d)
        if (i in excitatory_neurons and j not in excitatory_neurons) or (i not in excitatory_neurons and j in excitatory_neurons):
            return max(exc_connectivity_prob * np.exp(-d), inh_connectivity_prob * np.exp(-d))

    return 0

#randomly establishing connections between pairs of neurons in the
#neural network based on their indices and the probability of connection
connectivity_matrix = np.zeros((n_neurons, n_neurons))
for i in range(n_neurons):
    for j in range(n_neurons):
        if np.random.rand() < connection_probability(i, j):
            connectivity_matrix[i, j] = 1

#Simulation
# Simulation
def update_neuron(i, V, I, I_ext):
    # Update synaptic input
    dI = -I[i] / tau_I + np.sum(W_IE * connectivity_matrix[:n_excitatory, i] * (V[:n_excitatory] > threshold)) + \
         np.sum(W_EE * connectivity_matrix[:n_excitatory, i] * (V[:n_excitatory] > threshold)) + \
         np.sum(W_EI * connectivity_matrix[n_excitatory:, i] * (V[n_excitatory:] > threshold)) + \
         np.sum(W_II * connectivity_matrix[n_excitatory:, i] * (V[n_excitatory:] > threshold))
    I[i] += dI * dt

    # Update membrane potential
    if i in excitatory_neurons:
        dV = -V[i] / tau_exc + I[i] + I_ext
        Pr = Pr_exc
    else:
        dV = -V[i] / tau_inh + I[i] + I_ext
        Pr = Pr_inh
    V[i] += dV * dt

    # Determine if the neuron spikes
    if V[i] > threshold:
        V[i] = Pr
        return 1
    else:
        return 0

# Initialize variables
V = np.zeros(n_neurons)
I = np.zeros(n_neurons)
spike_times = [[] for _ in range(n_neurons)]

# Run simulation
for t in range(simulation_time):
    spikes = np.zeros(n_neurons)
    for i in range(n_neurons):
        spike = update_neuron(i, V, I,I0)
        spikes[i] = spike
        if spike:
            spike_times[i].append(t)
    if t % 100 == 0:
        print(f"Time: {t} ms")


# Visualizations and plots
# Plot grid
fig1, ax1 = plt.subplots( figsize=(8, 6))
grid_colors = ['red' if i in excitatory_neurons else 'blue' for i in range(n_neurons)]
grid_positions = [(i // grid_size, i % grid_size) for i in range(n_neurons)]
for pos, color in zip(grid_positions, grid_colors):
    ax1.scatter(pos[0], pos[1], c=color, s=10)
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_title('Neuronal Grid')
plt.tight_layout()
plt.show()

#Plot neuronal avalanches
fig2, ax2 = plt.subplots( figsize=(8, 6))
for i in range(n_neurons):
    ax2.plot(spike_times[i], [i] * len(spike_times[i]), 'k.', markersize=1)
ax2.set_xlabel('Time (ms)')
ax2.set_ylabel('Neuron Index')
ax2.set_title('Neuronal Avalanches')
plt.tight_layout()
plt.show()

# Color map of time since last spike
last_spike_times = np.zeros((grid_size, grid_size))
for i in range(n_neurons):
    x, y = i // grid_size, i % grid_size
    if spike_times[i]:
        last_spike_times[x, y] = simulation_time - spike_times[i][-1]
    else:
        last_spike_times[x, y] = simulation_time

fig3, ax3 = plt.subplots(figsize=(8, 6))
cmap = plt.cm.get_cmap('viridis')
im = ax3.imshow(last_spike_times, cmap=cmap)
cbar = fig3.colorbar(im, ax=ax3)
cbar.set_label('Time Since Last Spike (ms)')
ax3.set_xlabel('X')
ax3.set_ylabel('Y')
ax3.set_title('Time Since Last Spike')
plt.tight_layout()
plt.show()

# Print relevant statistics:
firing_rates = np.array([len(spikes) / (simulation_time / 1000) for spikes in spike_times])
print("Average Firing Rate (Excitatory):", np.mean(firing_rates[excitatory_neurons]))
print("Average Firing Rate (Inhibitory):", np.mean(firing_rates[inhibitory_neurons]))

isi_exc = [np.diff(spikes) for spikes in spike_times[:n_excitatory] if len(spikes) > 1]
isi_inh = [np.diff(spikes) for spikes in spike_times[n_excitatory:] if len(spikes) > 1]
if isi_exc:
    print("Mean Inter-Spike Interval (Excitatory):", np.mean(np.concatenate(isi_exc)))
else:
    print("Mean Inter-Spike Interval (Excitatory): No spikes")
if isi_inh:
    print("Mean Inter-Spike Interval (Inhibitory):", np.mean(np.concatenate(isi_inh)))
else:
    print("Mean Inter-Spike Interval (Inhibitory): No spikes")


''' ANALYSIS OF THE NEURONAL NETWORK '''
def simulate(I_ext):
    V = np.zeros(n_neurons)
    I = np.zeros(n_neurons)
    spike_times = [[] for _ in range(n_neurons)]
    avalanche_sizes = []
    activity_patterns = []

    # Run simulation
    for t in range(simulation_time):
        spikes = np.zeros(n_neurons)
        for i in range(n_neurons):
            spike = update_neuron(i, V, I,I_ext)
            spikes[i] = spike
            if spike:
                spike_times[i].append(t)
        activity_patterns.append(spikes)

        # Avalanche size calculation
        if np.sum(spikes) > 0:
            if len(avalanche_sizes) == 0 or avalanche_sizes[-1] != 0:
                avalanche_sizes.append(0)
            avalanche_sizes[-1] += np.sum(spikes)
        else:
            avalanche_sizes.append(0)

    return spike_times, avalanche_sizes, activity_patterns

# Simulation for subcritical, critical, and supercritical dynamics
I_values = [0, 0.1, 1.0]  # Adjust these values to explore different dynamical states
labels = ['Subcritical', 'Critical', 'Supercritical']
colors = ['blue', 'green', 'red']

fig, axs = plt.subplots(3, 3, figsize=(12, 12))

for idx, I_ext in enumerate(I_values):
    spike_times, avalanche_sizes, activity_patterns = simulate(I_ext)

    # Raster plot
    for i in range(n_neurons):
        axs[idx, 0].plot(spike_times[i], [i] * len(spike_times[i]), '.', markersize=1, color=colors[idx])
    axs[idx, 0].set_xlabel('Time (ms)')
    axs[idx, 0].set_ylabel('Neuron Index')
    axs[idx, 0].set_title(f'{labels[idx]} - Raster Plot')

    # Avalanche size distribution
    avalanche_sizes = [size for size in avalanche_sizes if size > 0]
    log_sizes = np.log10(avalanche_sizes)
    axs[idx, 1].hist(log_sizes, bins=20, color=colors[idx], alpha=0.7)
    axs[idx, 1].set_xlabel('Log10(Avalanche Size)')
    axs[idx, 1].set_ylabel('Count')
    axs[idx, 1].set_title(f'{labels[idx]} - Avalanche Size Distribution')

    # Entropy of activity patterns
    activity_patterns = np.array(activity_patterns)
    probs = np.mean(activity_patterns, axis=0)
    entropy_val = entropy(probs)
    axs[idx, 2].plot(probs, color=colors[idx])
    axs[idx, 2].set_xlabel('Neuron Index')
    axs[idx, 2].set_ylabel('Activity Probability')
    axs[idx, 2].set_title(f'{labels[idx]} - Entropy: {entropy_val:.2f}')

plt.tight_layout()
plt.show()

#Generating a live simulation of the spiking network:
#record spikes at each time step in 'all_spikes'
def simulate_network(I_ext):
    V = np.zeros(n_neurons)
    I = np.zeros(n_neurons)
    spike_times = [[] for _ in range(n_neurons)]
    all_spikes = []

    for t in range(simulation_time):
        spikes = np.zeros(n_neurons)
        for i in range(n_neurons):
            spike = update_neuron(i, V, I, I_ext)
            spikes[i] = spike
            if spike:
                spike_times[i].append(t)
        all_spikes.append(spikes)

    return spike_times, all_spikes

spike_times, all_spikes = simulate_network(I0)

fig, ax = plt.subplots()
grid_positions = [(i // grid_size, i % grid_size) for i in range(n_neurons)]
scat = ax.scatter([pos[0] for pos in grid_positions], [pos[1] for pos in grid_positions], c='blue')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('Neuronal Grid Dynamics')

#updating the neuron colors based on whether they spiked at the current frame (time step)
def update_animation(frame):
    spikes = all_spikes[frame]
    colors = ['red' if spike else 'blue' for spike in spikes]
    scat.set_color(colors)
    ax.set_title(f'Time: {frame} ms')
    return scat,

ani = animation.FuncAnimation(fig, update_animation, frames=simulation_time, interval=20, blit=True, repeat=False)
plt.show()


# Function to fit power-law distribution
def power_law(x, alpha, C):
    return C * (x ** (-alpha))

# Function to plot avalanche size distribution and power-law fit
def plot_avalanche_distribution(avalanche_sizes):
    # Calculate histogram of avalanche sizes
    counts, bins = np.histogram(avalanche_sizes, bins='auto', density=True)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    nonzero_counts = counts[counts > 0]
    nonzero_bin_centers = bin_centers[counts > 0]

    # Fit power-law distribution to the data
    popt, _ = curve_fit(power_law, nonzero_bin_centers, nonzero_counts)

    # Plot distribution and power-law fit
    plt.loglog(nonzero_bin_centers, nonzero_counts, 'bo', label='Avalanche Sizes')
    plt.loglog(nonzero_bin_centers, power_law(nonzero_bin_centers, *popt), 'r-', label=f'Power Law Fit (alpha={popt[0]:.2f})')
    plt.xlabel('Avalanche Size')
    plt.ylabel('Probability Density')
    plt.title('Avalanche Size Distribution')
    plt.legend()
    plt.show()

''' take a look!!! 
# Assuming avalanche_sizes contains the list of avalanche sizes from your simulation
#avalanche_sizes = avalanche_sizes  # Replace [...] with your actual list of avalanche sizes

# Plot avalanche size distribution and fit power-law
plot_avalanche_distribution(avalanche_sizes)
'''
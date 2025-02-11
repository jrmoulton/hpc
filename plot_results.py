import matplotlib.pyplot as plt
import numpy as np

# Read the benchmark results
data = np.loadtxt('benchmark_results.txt', delimiter=',')
threads = data[:, 0]
times = data[:, 1]

# Calculate speedup relative to single thread
speedup = times[0] / times

# Create the plot
plt.figure(figsize=(10, 6))

# Plot speedup
plt.plot(threads, speedup, 'bo-', label='Actual speedup')
# Plot ideal speedup
plt.plot(threads, threads, 'r--', label='Ideal linear speedup')

plt.xlabel('Number of Threads')
plt.ylabel('Speedup')
plt.title('Strong Scaling Performance')
plt.grid(True)
plt.legend()

# Save the plot
plt.savefig('scaling_results.png')
plt.close()

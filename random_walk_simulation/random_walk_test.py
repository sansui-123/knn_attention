import torch
import numpy as np
from random_walk_simulation.random_walk import approximate_product, simulate_random_walks

n = 10000
d = 5
B = 10
num_samples = 10000

# Note that we need all real numbers to be float64 to avoid numerical issues.

# Create a random vector of shape (n,1), its entries are real numbers drawn uniformly at random from [-B,B]
y = torch.rand(n).to(torch.float64)
y = y * 2 * B - B

# Create a random tensor Q,K of shape (n,d), its entries are drawn uniformly at random from [-B,B]
Q = torch.rand(n,d).to(torch.float64)
Q = Q * 2 * B - B
K = torch.rand(n,d).to(torch.float64)
K = K * 2 * B - B

# If P = softmax(Q @ K^T), then calculate P^T y
P = torch.softmax(Q @ K.t(), dim=0)
Py = P.t() @ y

# Now we want to test if we can calculate this via MCMC methods.
estimated_Py = approximate_product(Q, K, y, num_samples)

# Print the mean absolute error.
print("Mean absolute error:", torch.mean(torch.abs(Py - estimated_Py)).item())

# Find the mean relative error, ignoring values close to zero.
mean_relative_error = 0
for i in range(n):
    if Py[i] > 1:
        mean_relative_error += torch.abs(Py[i] - estimated_Py[i]) / torch.abs(Py[i]).to(torch.float64)

mean_relative_error /= n
print("Mean relative error:", mean_relative_error)

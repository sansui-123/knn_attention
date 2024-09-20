#
# This file contains functions to calculate matrix vector products of the form
# P^T x where P is a stochastic matrix of shape n x n and x is a vector of shape n x 1.
# 
# The product is approximated using Markov Chain Monte Carlo methods.
#

import torch
import math
import numpy as np

# 
# This function calculates the product P^T x where P is a stochastic matrix of shape n x n
# and x is a vector of shape n x 1. The product is approximated using Markov Chain Monte Carlo methods.
#
# Input:
#   - y: Vector of shape n x 1 -> This is the initial distribution over [n].
#   - Q: A matrix of shape n x d -> This is the query matrix.
#   - K: A matrix of shape n x d -> This is the key matrix.
#        Note that our stochastic matrix P is given by P = softmax(Q @ K^T).
#   - num_samples: Number of samples to take.
#
# Output: a vector of shape n x 1 -> This is the approximation of P^T x.
#
def simulate_random_walks(y, Q, K, num_samples):

    # Assert that y is a proper probability distribution
    # assert math.isclose(torch.sum(y).item(), 1.0, rel_tol=1e-3), "y is not a proper probability distribution."

    # First, flatten y to a 1D array.
    y = y.view(-1)

    # Get the number of states
    states = torch.arange(len(y))
    counts = torch.zeros_like(y)

    for _ in range(num_samples):
        # Sample initial state based on the initial distribution
        initial_state = np.random.choice(states, p=y.detach().numpy())

        # Sample the next state based on the transition probabilities from the initial state
        next_state = np.random.choice(states, p=torch.softmax(Q[initial_state] @ K.t(), dim=0).detach().numpy())

        # Count the visit to the next state
        counts[next_state] += 1

    # Estimate the distribution
    estimated_distribution = counts / num_samples
    return estimated_distribution


# 
# This function calculates the product P^T y where P is a stochastic matrix of shape n x n.
# The product is approximated using Markov Chain Monte Carlo methods.
# 
# The input vector y does not have to be positive or normalized.
#
def approximate_product(Q, K, y, num_samples):
    # Find the minimum entry of y.
    M = torch.min(y)

    if M >= 0:
        M = 0
    else:
        M = -M

    # Add M to each entry of y.
    y_prime = y + M

    # Normalize y_prime.
    normalization = torch.sum(y_prime)
    y_prime /= normalization

    # Compute the product P^T y_prime using MCMC methods. 
    Py_prime = simulate_random_walks(y_prime, Q, K, num_samples)

    # Now we need to un-normalize Py_prime.
    Py = Py_prime * normalization

    if M == 0:
        return Py

    # Consider PM to be the product P^T M*1^n, where 1^n is the vector of all ones.
    # We can estimate PM by sampling from the uniform distribution, again using MCMC methods.
    M_vec = M * torch.ones_like(y)
    M_vec_normalization = M * len(y)
    M_vec /= M_vec_normalization
    PM = simulate_random_walks(M_vec, Q, K, num_samples)

    # Un-normalize PM.
    PM *= M_vec_normalization

    # Assemble the final estimate.
    Py_estimate = Py - PM

    return Py_estimate
# Description: This file contains the forward pass functions for the attention mechanism.
# We implement kNN attention in a vectorized manner.

import torch
from softmax_expectation.softmax_expectation import softmax_expectation_estimation_faster
from softmax_expectation.topk import topk
import numpy as np
import math

# This function calculates the attention mechanism in the forward pass.
# It uses the softmax expectation approximation method.
def attn_forward_unvectorized(Q, K, V, epsilon=0.01, delta=0.01):

    # Calculate the number of rows in Q.
    n = Q.shape[0]

    # Calculate the number of columns in V.
    d = V.shape[1]

    # Calculate the top sqrt(n) indices of Q[i] @ K[j]^T.
    k = 0.75 * n # int(n ** 0.5)
    scores, S = topk(Q, K, k)

    S = torch.tensor(S, dtype=torch.int64)
    scores = torch.tensor(scores, dtype=torch.float32)

    # Initialize the output tensor.
    output = torch.zeros_like(V, dtype=torch.float32)

    MM = 20

    # For all rows in Q...
    for i in range(n):
        # For all columns in V...
        denom = torch.sum(torch.exp(scores[i, :]-MM))
        for j in range(d):
            # Approximate the expected value of the value vector.
            
            def f(Q, K, V, dO, index, i, j=None):
                return V[index, j]
            
            inputs = (V, None, None, j)

            output[i, j] = softmax_expectation_estimation_faster(
                Q, K, i, \
                f, \
                inputs, \
                S[i,:].tolist(), \
                scores[i, :].tolist(), \
                denom=denom, \
                epsilon=epsilon, delta=delta).item()

    return output

# This function calculates the attention mechanism in the forward pass.
# Inputs:
# - Q: A tensor of shape (b,h,n,d) containing the query vectors.
# - K: A tensor of shape (b,h,n,d) containing the key vectors.
# - V: A tensor of shape (b,h,n,d) containing the value vectors.
#
# Note that b is the batch size, h is the number of heads, 
# n is the sequence length, and d is the dimension of the vectors.
#
# Outputs:
# - A tensor of shape (b,h,n,d) containing the output vectors.
def attn_forward_batched(Q, K, V, k, epsilon=0.01, delta=0.01):
        
    B,H,N,D = Q.shape

    # -- VECTORIZED CODE -- #
    #
    output = torch.zeros(B,H,N,D, dtype=torch.float32)
    for b in range(B):
        for h in range(H):

            # Get the top k indices of Q[b,h] @ K[b,h]^T and the scores.
            scores, S = topk(Q[b,h,:,:], K[b,h,:,:], k, masking=True)
            scores = scores * (1 / math.sqrt(D))

            # Calculate the denominator.
            M = torch.max(scores, dim=1)[0]
            denom = torch.sum(torch.exp(scores - M.unsqueeze(1)), dim=1).unsqueeze(1)

            # Calculate the numerator.
            # Vbh[S].shape = (N,k,D)
            numerator = torch.bmm(torch.exp(scores - M.unsqueeze(1)).unsqueeze(1), V[b,h,S]).squeeze()

            output[b,h] = numerator / denom

    return output

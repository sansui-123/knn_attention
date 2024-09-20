# This file contains the unoptimized version of the forward pass functions.
# It is kept for reference and comparison purposes.
# See forward_pass.py for the optimized version.

import numpy as np
import torch
import time
from memory_profiler import profile

#
# Approximating the expected value of a value vector v with underlying distribution
# p(i) = softmax(q^T k_i) / Z, where Z is the partition function
# 
# Parameters:
#   q is the query vector: 1 x d
#   K is the matrix of key vectors: n x d
#   v is the value vector: 1 x n
#   k is the number of top-k elements to consider
#   l is the number of samples to draw from the remaining elements
#   topk_indices_func is a function that returns the top-k indices of the attention scores
#   B is the maximum value of the q,k,v elements. We use this to avoid numerical instability.
# 
def approximate_softmax_expectation(q, K, v, \
                                    k, l, \
                                    topk_indices_func, \
                                    B, \
                                    lsh_objects=None, \
                                    index=None,
                                    topk_scores=None, \
                                    topk_indices=None, \
                                    query_idx = None):
    n, d = K.size()

    # Find the top-k indices of the attention scores
    if query_idx == None:
        scores, indices = topk_indices_func(q, K, k, B, lsh_objects=lsh_objects, index=index)
    else:
        scores = topk_scores[query_idx]
        indices = topk_indices[query_idx]

    # From the n-k remaining elements, draw l samples.
    random_indices = []
    while len(random_indices) < l:
        index = np.random.randint(n)
        if index not in indices:
            random_indices.append(index)

    # Now we'll evaluate the partition function and the expectation separately.

    approx_partition = 0
    approx_expectation = 0
    for index in random_indices:
        # Calculate the attention score for the remaining elements
        attention_score = torch.exp(torch.dot(q, K[index]) / d - B*B)

        # Add the attention score to the partition function
        approx_partition += attention_score

        # Add the attention score times the value to the expectation
        approx_expectation += attention_score * v[index]

    approx_partition *= ((n-k) / l)
    approx_expectation *= ((n-k) / l)

    approx_partition += scores.sum()
    approx_expectation += torch.sum(scores * v[indices])

    # Return the approximate softmax partition function.
    return approx_expectation / approx_partition


# 
# Approximates the attention output using sampling.
# 
# Parameters:
#   Q is the matrix of query vectors: n x d
#   K is the matrix of key vectors: n x d
#   V is the matrix of value vectors: n x d
#   k is the number of top-k elements to consider
#   l is the number of samples to draw from the remaining elements
#   topk_indices_func is the function to find the top-k indices
#
def sampling_attention(Q, K, V, k, l, topk_indices_func, B, lsh_objects=None, index=None):

    n,d = Q.size()

    # Divide all the key vectors by d.
    K = K / d

    output = torch.zeros_like(V)

    # For all rows in Q...
    for i in range(Q.size(0)): # n
        # For all columns in V...
        for j in range(V.size(1)): # d
            # Approximate the expected value of the value vector

            # Time this function
            # import time
            # start_time = time.time()
            output[i, j] = approximate_softmax_expectation(Q[i], K, V[:,j], k, l, topk_indices_func, B, lsh_objects, index)
            # print("Time taken for expectation: ", time.time() - start_time)

    return output

# Sampling attention with clustering
# Version 1: Use the top-k indices and scores as input
def sampling_attention_clustering_v1(Q, K, V, k, l, topk_scores, topk_indices, B):
    n,d = Q.size()

    # Divide all the key vectors by d.
    K = K / d

    output = torch.zeros_like(V)

    # For all rows in Q...
    for i in range(Q.size(0)): # n

        if i % 1000 == 0:
            print("Processing query ", i)

        # For all columns in V...
        for j in range(V.size(1)): # d
            # Approximate the expected value of the value vector

            # Time this function
            # import time
            # start_time = time.time()
            output[i, j] = approximate_softmax_expectation(Q[i], K, V[:,j], \
                                                           k, l, \
                                                           None, B,
                                                           None, 
                                                           None,
                                                           topk_scores,
                                                           topk_indices,
                                                           i)
            # print("Time taken for expectation: ", time.time() - start_time)

    return output


#
# Approximating the expected value of a value vector v with underlying distribution
# p(i) = softmax(q^T k_i) / Z, where Z is the partition function
#
# Version 2: Use the same samples for a given query for all the value vectors.
# 
# Parameters:
#   q is the query vector: 1 x d
#   K is the matrix of key vectors: n x d
#   V is the value vector matrix: n x d
#   k is the number of top-k elements to consider
#   l is the number of samples to draw from the remaining elements
#   topk_indices_func is a function that returns the top-k indices of the attention scores
#   B is a normalizer for the attention scores: 1 x n
#
#   topk_scores is the top-k attention scores for all the queries
#   topk_indices is the top-k indices for all the queries
# 
# Returns:
#   The approximate expected value of the value vectors v: 1 x d
def approximate_softmax_expectation_v2(q, K, V, \
                                        k, l, \
                                        topk_indices_func, \
                                        B, \
                                        lsh_objects=None, \
                                        index=None,
                                        topk_scores=None, \
                                        topk_indices=None, \
                                        query_idx = None):
    n, d = K.size()

    # Find the top-k indices of the attention scores
    if query_idx == None:
        scores, indices = topk_indices_func(q, K, k, B, lsh_objects=lsh_objects, index=index)
    else:
        scores = topk_scores[query_idx] # k x 1
        indices = topk_indices[query_idx] # k x 1

    # From the n-k remaining elements, draw l samples.
    # random_indices = []
    # while len(random_indices) < l:
    #     index = np.random.randint(n)
    #     if index not in indices:
    #         random_indices.append(index)

    # Now we'll evaluate the partition function and the expectation separately.
    approx_partition = 0
    approx_expectation = torch.zeros(d)
    
    i = 0
    while i < l:

        index = np.random.randint(n)

        if index in indices:
            continue

        i = i + 1

        # Calculate the attention score for the remaining elements
        attention_score = torch.exp(torch.dot(q, K[index]) - B[query_idx])

        # Add the attention score to the partition function
        approx_partition += attention_score

        # Add the attention score times the value to the expectation
        approx_expectation += attention_score * V[index, :]

    approx_partition *= ((n-k) / l)
    approx_expectation *= ((n-k) / l)

    approx_partition += scores.sum()
    approx_expectation += torch.sum(scores.view(-1, 1) * V[indices, :], dim=0)

    # Return the approximate softmax partition function.
    return approx_expectation / approx_partition


# Clustering sampling attention
# Version 2: Use the same samples for a given query for all the value vectors.
@profile
def sampling_attention_clustering_v2(Q, K, V, k, l, topk_scores, topk_indices, B):
    n,d = Q.size()

    # Divide all the key vectors by d.
    K = K / d

    output = torch.zeros_like(V)

    # For all rows in Q...
    for i in range(Q.size(0)): # n

        if i % 1000 == 0:
            print("Processing query ", i)

        # Approximate the expected value of the value vector

        # Time this function
        # import time
        # start_time = time.time()
        # Takes around O(l) time.
        output[i, :] = approximate_softmax_expectation_v2(Q[i], K, V, \
                                                          k, l, \
                                                          None, B, \
                                                          None, \
                                                          None, \
                                                          topk_scores, \
                                                          topk_indices, \
                                                          i)
        # print("Time taken for expectation: ", time.time() - start_time)

    return output
from .concentric_lsh import angularLSH
import torch
import math
import numpy as np

# We will create a sequence of concentric LSH objects with increasing r values.
# !! Note !! - c requires tweaking! It is a hyperparameter that can be adjusted.
def topk_indices_lsh_preprocessing(K, B, verbose=False, c=1):
    current_s = 0

    # We will create a sequence of LSH objects with increasing r values.
    # We cannot go all the way to 1, because then we get instability issues.
    lsh_objects = []
    while current_s < B*B - c:
        lsh_objects.append(angularLSH(K, s=current_s, b=current_s+c, B=B, verbose=verbose))
        current_s += c

    return lsh_objects

#
# This function finds the top-k inner products q^T k_i for a query q and a set of 
# keys K. It does so in O(k) time by using the concentric circle LSH idea.
#
# Requires topk_indices_lsh_preprocessing() to be run first.
# 
# Parameters:
#   q is the query vector: 1 x d
#   K is the matrix of key vectors: n x d 
#       -> they have been divided by d already.
#   k is the number of top-k elements to consider
#       -> set to something like n^{2/3}
#   B is the maximum value of the q,k,v elements. 
#       -> We use this to avoid numerical instability.
#   lsh_objects is a list of LSH objects with increasing s values.
#       -> We use these to find the top-k keys.
#
# Returns:
#   attention_scores is a tensor of size k containing the attention scores.
#   keys is a list of the indices of the top-k keys.
#
def topk_indices_fast_lsh(q, K, k, B, lsh_objects, index, verbose=False):

    n, d = K.size()

    # We add a zero to the query vector to make it d+1.
    q_normalized = q.clone()
    q_normalized = torch.cat((q_normalized, torch.tensor([0.0])))

    keys_before = set()
    keys_after = set()

    # Find the first LSH object where at least k keys are hashed to the same bucket as the query vector.
    for i in range(len(lsh_objects)):
        if i > 0:
            keys_before = keys_after

        keys_after, num_keys = lsh_objects[i].query_bucket_size(q_normalized, 2 * k)

        if verbose:
            print("Lsh object", i, "has", num_keys, "keys. These are: ", keys)
    
        if num_keys >= k:
            break

    # Now we have two LSH objects, so that the first one has less than k keys
    # hashed to the same bucket as the query vector, and the second one has at least k keys.
    # We will take all the keys from the first one and supplement with keys from the second one.

    if verbose:
        print(keys_before)
        print(keys_after)

    keys = keys_before
    for key in keys_after:
        if key not in keys:
            keys.add(key)

        if len(keys) >= k:
            break

    # Now we have the top-k keys. We will calculate the attention scores for these keys.
    attention_scores = torch.zeros(len(keys))
    for i, key in enumerate(keys):
        attention_scores[i] = torch.dot(q, (K[key, :] / d)) - B*B
        attention_scores[i] = torch.exp(attention_scores[i])

    return attention_scores, list(keys)
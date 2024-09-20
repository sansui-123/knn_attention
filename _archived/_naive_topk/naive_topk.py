import torch

# 
# Find top-k indices of inner products q^T k_i for every query q and set of keys K
# This is done naively here, in O(n^2) time
# 
def topk_indices_naive(Q, K, k, B, lsh_objects=None, index=None):
    n, d = K.size()

    # Calculate the dot product of Q and K
    attention_scores = torch.matmul(Q, K.transpose(-2, -1)) 

    # Normalize the attention scores by dividing by d.
    attention_scores = attention_scores / d

    # Apply the exponential function to the attention scores
    attention_scores = torch.exp(attention_scores - B * B)
    
    # Find the top-k indices of the attention scores
    topk_scores, topk_indices = torch.topk(attention_scores, k, dim=-1)
    
    return topk_scores, topk_indices
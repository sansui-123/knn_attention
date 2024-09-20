import torch
from memory_profiler import profile

# @profile
def calculate_attention(Q, K, V, retain_grad=False):
    n, d = Q.size()

    # Calculate the dot product of Q and K
    attention_scores = torch.matmul(Q, K.transpose(0, 1))

    # Normalize the attention scores by dividing by d.
    attention_scores = attention_scores # / d **0.5
    
    # Apply softmax to get the attention weights
    attention_weights = torch.softmax(attention_scores, dim=-1)
    
    # Multiply the attention weights with the value vectors
    attention_output = torch.matmul(attention_weights, V)
    
    if retain_grad:
        attention_output.requires_grad = True
    
    return attention_output

def calculate_attention_batched(Q,K,V):
    B,H,N,D = Q.shape
    # Calculate the dot product of Q and K
    attention_scores = torch.matmul(Q, K.transpose(-2, -1)) # B x H x N x N

    # Mask the attention scores (lower triangular)
    mask = torch.tril(torch.ones(N, N)).unsqueeze(0).unsqueeze(0)
    attention_scores = attention_scores.masked_fill(mask == 0, float('-inf'))

    # Normalize the attention scores by dividing by d.
    attention_scores = attention_scores / (D **0.5)
    
    # Apply softmax to get the attention weights
    attention_weights = torch.softmax(attention_scores, dim=-1)
    
    # Multiply the attention weights with the value vectors
    attention_output = torch.matmul(attention_weights, V)
    
    return attention_output
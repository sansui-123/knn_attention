import torch
import time
import numpy as np
from naive_topk import topk_indices_naive
from forward_pass import forward_pass_unoptimized
from naive_attention import naive_attention
import math

n, d = 200, 30

# Generate 50x50 random matrices with elements in range [-10, 10]
Q = torch.randint(-10, 11, (n, d)).float()
K = torch.randint(-10, 11, (n, d)).float()
V = torch.randint(-10, 11, (n, d)).float()

k = l = 50

# Call the sampling_attention function.
# Measure the time it takes to run the function.
start_time = time.time()
attention_output = forward_pass_unoptimized.sampling_attention(Q, K, V, k,l, topk_indices_naive, B=10.0)
end_time = time.time()
print("Time taken for approximate attention:", end_time - start_time)

# Compare with the exact attention output
# Calculate the time taken to run the calculate_attention function.
start_time = time.time()
exact_attention_output = naive_attention.calculate_attention(Q, K, V, B=10.0)
end_time = time.time()
print("Time taken for exact attention:", end_time - start_time)

# Print the mean absolute error
print("Mean error: ", torch.mean(torch.abs(attention_output - exact_attention_output)))
print("Max error: ", torch.max(torch.abs(attention_output - exact_attention_output)))
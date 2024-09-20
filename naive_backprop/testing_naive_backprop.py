import torch
from torch import nn
from naive_attention import naive_attention
from naive_backprop import naive_backprop

# This file will test the correctness of self-attention backprop implementation.
# First, suppose we have a loss function f, which will be the cross-entropy loss.
loss = nn.CrossEntropyLoss()

# Vocab size. This is the number of classes in the classification task.
vocab_size = 10

# Embedding dimension
d = 5

# Input sequence length
N = 10

# Q,K,V matrices - input to self-attention
Q = torch.randn(N, d, requires_grad=True) * 10 + 20
Q.retain_grad()
K = torch.randn(N, d, requires_grad=True) * 10 + 20
K.retain_grad()
V = torch.randn(N, d, requires_grad=True) * 10 + 20
V.retain_grad()

# Calculate naive attention
O = naive_attention.calculate_attention(Q, K, V) # N x d
O.retain_grad()

# Create a linear layer
W = nn.Linear(d, vocab_size, bias=False) # d x V
L = W(O) # N x V
L.retain_grad()

# Calculate the loss. This is the average cross entropy loss for all N positions in the 
# input sequence.
T = torch.randint(0, vocab_size, (N,)) # N x 1
loss = loss(L, T)
print(f"Loss = {loss.item()}")

# Calculate the gradients of the loss w.r.t. W and O.
loss.backward()

# Sanity check: dO = dL * W.T
if torch.max(torch.abs(O.grad - L.grad.mm(W.weight))) > 1e-3:
    print("Gradients of O and L are not consistent.")

# Calculate the gradient with respect to V:
dV = naive_backprop.grad_V(Q, K, V, O.grad)
if torch.max(torch.abs(dV - V.grad)) > 0.001:
    print("Gradients of V are not correct.")

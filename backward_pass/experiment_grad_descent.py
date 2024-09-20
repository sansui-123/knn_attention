# Experiment: Gradient on a non-convex loss function using our method vs the naive method.
# Loss function: Cross entropy loss

import torch
from torch import nn
from naive_attention import naive_attention
from naive_backprop import naive_backprop
from backward_pass.backward_pass import fast_grad_V, fast_grad_Q, fast_grad_K
import matplotlib.pyplot as plt
import matplotlib as mpl
import torch.nn.functional as F

mpl.rcParams['figure.dpi']= 300

# Embedding dimension
d = 3

# Input sequence length
N = 100

lr = 0.5

# Q,K,V matrices - input to self-attention
Q = torch.randn(N, d) * 2
Q.requires_grad = True
K = torch.randn(N, d) * 2
K.requires_grad = True
V = torch.randn(N, d) * 2
V.requires_grad = True

Q_copy_fast = Q.clone().detach().requires_grad_(True)
K_copy_fast = K.clone().detach().requires_grad_(True)
V_copy_fast = V.clone().detach().requires_grad_(True)

targets = torch.randint(0, 2, (N, )).float()
target_weights = torch.randn(d, 1)

# We'll run gradient descent on the loss function.
# We'll compare the descent with the gradients calculated using our method, vs
# the gradients calculated using the naive method.
iteration = 0
num_iterations = 1500
losses_fast = []
losses = []
while iteration != num_iterations:
    iteration += 1
    print(f"Iteration: {iteration}")

    # Calculate naive attention
    O_fast = naive_attention.calculate_attention(Q_copy_fast, K_copy_fast, V_copy_fast) # N x d
    O_fast.retain_grad()

    O = naive_attention.calculate_attention(Q, K, V) # N x d
    O.retain_grad()

    output = torch.mm(O, target_weights).squeeze() # N x 1
    output_fast = torch.mm(O_fast, target_weights).squeeze() # N x 1

    # Calculate the loss
    # Loss: squared error between the output and the target
    # loss = torch.mean((output - targets.float()) ** 2)
    # loss_fast = torch.mean((output_fast - targets.float()) ** 2)
    loss = F.binary_cross_entropy_with_logits(output, targets.float())
    loss_fast = F.binary_cross_entropy_with_logits(output_fast, targets.float())

    losses_fast.append(loss_fast.item())
    losses.append(loss.item())

    # Calculate the gradients of the loss 
    loss.backward(retain_graph=True)
    loss_fast.backward(retain_graph=True)

    # Approximate the gradient with respect to V:
    dV_fast = fast_grad_V(Q_copy_fast, K_copy_fast, V_copy_fast, O_fast.grad, epsilon=0.05)

    # Print the mean absolute error:
    # print(f"dV: Mean absolute error: {torch.mean(torch.abs(dV_fast - V.grad)).item()}")

    # Approximate the gradient with respect to Q.
    # First, clone the input matrices.
    Q_copy = Q.clone().detach().requires_grad_(False)
    K_copy = K.clone().detach().requires_grad_(False)
    V_copy = V.clone().detach().requires_grad_(False)
    dQ_fast = fast_grad_Q(Q_copy, K_copy, V_copy, O.grad, epsilon=0.05, delta=0.1)

    # Print the mean absolute error:
    # print(f"dQ: Mean absolute error: {torch.mean(torch.abs(dQ_fast - Q.grad)).item()}")

    # Update the Q, V matrices
    with torch.no_grad():
        Q -= lr * Q.grad
        V -= lr * V.grad
        Q.grad.zero_()
        V.grad.zero_()

        Q_copy_fast -= lr * dQ_fast
        V_copy_fast -= lr * dV_fast

# Plot the loss and the loss from the fast gradients
plt.plot(losses_fast)
plt.plot(losses)
plt.legend(['Fast gradients', 'Naive gradients'])
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title(r'Loss: Cross Entropy')

plt.savefig('backward_pass/assets/experiment_cross_entropy.png')
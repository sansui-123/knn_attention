import torch


# Naive self-attention backpropagation.
# Suppose we have some loss function f.
# Input:
#   - Q: Query matrix of shape: N x d
#   - K: Key matrix of shape: N x d
#   - V: Value matrix of shape: N x d
#   - dO: Derivative of f with respect to the output of the self-attention layer of shape: N x d
# 
# Output:
#   - dV: Derivative of f with respect to V of shape: N x d
def grad_V(Q, K, V, dO):
    dV = torch.zeros_like(V)

    # Formula for derivative of f with respect to V:
    # dV = P^T * dO
    # P = softmax(QK^T) -> N x N

    # Compute P
    P = torch.mm(Q, K.t())
    P = torch.nn.functional.softmax(P, dim=1)

    # Compute dV
    dV = torch.mm(P.t(), dO)

    return dV

def dP(dO, V, i, j):
    return dO[i,:] @ V[j,:]

def P(Q, K, i,j):
    distribution = torch.softmax(Q[i,:] @ K.T, dim=0)
    return distribution[j]

def dPP(Q,K,V,dO,i):
    n,d = Q.shape
    ip1 = torch.tensor([dP(dO,V,i,k) for k in range(n)])
    ip2 = torch.tensor([P(Q,K,i,k) for k in range(n)])
    ip = ip1 @ ip2
    
    return ip

def grad_Q(Q, K, V, dO):
    dQ = torch.zeros_like(Q)

    n, d = Q.shape

    for i in range(n):
        for j in range(d):
            for k in range(n):
                dQ[i,j] += P(Q,K,i,k) * (dP(dO,V, i, k) - dPP(Q,K,V,dO,i)) * K[k,j]

    return dQ

def grad_K(Q, K, V, dO):
    dK = torch.zeros_like(K)

    n, d = Q.shape

    for i in range(n):
        for j in range(d):
            for k in range(n):
                dK[i,j] += P(Q,K,k,i) * (dP(dO,V,k, i) - dPP(Q,K,V,dO,k)) * Q[k,j]
    
    return dK
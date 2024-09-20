import torch
import math
import numpy as np
import faiss
import time

# 
# Find the top k indices of Q[i] @ K[j]^T for j in [n] and all i in [n]
#
# Input:
#   - Q: A matrix of shape n x d -> This is the query matrix.
#   - K: A matrix of shape n x d -> This is the key matrix.
#   - k: The number of indices to return.
#
# Output:
#   - The top k indices of Q[i] @ K[j]^T.
#
def topk(Q, K, k, masking=False):
	n, d = Q.shape
    # We will use the Faiss library to perform the clustering.
    
	K_normalized = K.clone().detach().requires_grad_(False)
	Q_clone = Q.clone().detach().requires_grad_(False)

    # Build a FAISS Flat index:
	if masking == False:
		index = faiss.IndexFlatIP(d)

		# Add the normalized key vectors to the index
		index.add(K_normalized.cpu().float().numpy())

		# Search for each query vector in Q in the index for
		# its k nearest neighbors.
		# This takes O(nk) time.
		scores, I = index.search(Q_clone.cpu().float().numpy(), k)
	else:
		# If masking is turned on, each query vector Q[i] only looks for top-k
		# keys K[j] such that j < i.

		index = faiss.IndexFlatIP(d)
		for i in range(n):
			index.add(K_normalized[i].cpu().float().numpy().reshape(1,-1))
			scores_i, I_i = index.search(Q_clone[i].cpu().float().numpy().reshape(1,-1), k)
			scores_i = torch.tensor(scores_i)
			I_i = torch.tensor(I_i)
			if i == 0:
				scores = scores_i
				I = I_i
			else:
				scores = torch.cat((scores, scores_i), dim=0)
				I = torch.cat((I, I_i), dim=0)
	return scores, I


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
def attn_forward_batched(Q, K, V):
        
	B,H,N,D = Q.shape
	
	k = 5 # int(N ** 0.125) # Experiments show this is good.
	
	output = torch.zeros(B,H,N,D, dtype=torch.float32).to("cuda")
	for b in range(B):
		for h in range(H):

			scores, S = topk(Q[b,h,:,:], K[b,h,:,:], k, masking=False)
			scores = torch.tensor(scores, dtype=torch.float32).to("cuda")
			scores *= (1 / math.sqrt(D))
			S = torch.tensor(S, dtype=torch.int64).to("cuda")

			M = torch.max(scores, dim=1)[0]
			M = M.to("cuda")
			denom = torch.sum(torch.exp(scores - M.unsqueeze(1)), dim=1).unsqueeze(1)

            # V[b,h,S].shape = (N,k,D)
			numerator = torch.bmm(torch.exp(scores - M.unsqueeze(1)).unsqueeze(1), V[b,h,S]).squeeze()
			output[b,h] = numerator / denom

	return output
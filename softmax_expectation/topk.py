import torch
import faiss

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
def topk(Q, K, k, masking=False):
    n, d = Q.shape
    # We will use the Faiss library to perform the clustering

    # Build a FAISS Flat index:
    if masking == False:
        index = faiss.IndexFlatIP(d)

        K_normalized = K.clone()

        # Add the normalized key vectors to the index
        index.add(K_normalized.numpy().astype('float32'))

        # Search for each query vector in Q in the index for
        # its k nearest neighbors.
        # This takes O(nk) time.
        scores, I = index.search(Q.numpy().astype('float32'), k)
        scores = torch.tensor(scores)
        I = torch.tensor(I)
    else:
        # If masking is turned on, each query vector Q[i] only looks for top-k
        # keys K[j] such that j < i.

        index = faiss.IndexFlatIP(d)
        for i in range(n):
            index.add(K[i].reshape(1, -1).numpy().astype('float32'))
            scores_i, I_i = index.search(Q[i].reshape(1, -1).numpy().astype('float32'), k)
            scores_i = torch.tensor(scores_i)
            I_i = torch.tensor(I_i)
            if i == 0:
                scores = scores_i
                I = I_i
            else:
                scores = torch.cat((scores, scores_i), dim=0)
                I = torch.cat((I, I_i), dim=0)

    return scores, I
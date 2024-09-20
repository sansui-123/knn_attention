import torch
import faiss
from memory_profiler import profile

# 
# Function for preprocessing the input data for clustering_topk
# We just use a flat L2 index for the key vectors.
#
def clustering_topk_preprocessing_flatl2(K, B, verbose=False):
    n,d = K.size()

    # The key vectors can have different lengths, so we will normalize them
    # by adding an extra dimension to each key vector.
    K_normalized = K.clone()
    K_normalized = torch.cat((K_normalized, torch.sqrt(d*B*B - torch.sum(K_normalized**2, dim=1, keepdim=True))), dim=1)

    # The dimension of the key vectors has increased by 1.
    d = d + 1

    # We will use the Faiss library to perform the clustering.

    # Build a FAISS Flat index:
    index = faiss.IndexFlatL2(d)

    # Add the normalized key vectors to the index
    index.add(K_normalized.numpy())

    return index

# 
# Function for preprocessing the input data for clustering_topk
# We just use an IVFFlat index for the key vectors.
#
def clustering_topk_preprocessing_ivff_index(K, B, voronoi_cells=100, nprobe=10, verbose=False):
    n,d = K.size()

    # The key vectors can have different lengths, so we will normalize them
    # by adding an extra dimension to each key vector.
    K_normalized = K.clone()
    K_normalized = torch.cat((K_normalized, torch.sqrt(d*B*B - torch.sum(K_normalized**2, dim=1, keepdim=True))), dim=1)

    # The dimension of the key vectors has increased by 1.
    d = d + 1

    # We will use the Faiss library to perform the clustering.
    # We will use the IVFFlat index for the key vectors.

    nlist = voronoi_cells
    quantizer = faiss.IndexFlatL2(d)  # the other index
    index = faiss.IndexIVFFlat(quantizer, d, nlist)

    assert not index.is_trained
    index.train(K_normalized.numpy())
    assert index.is_trained

    # Configure the number of neigboring voronoi cells to search
    # This is called the nprobe parameter.
    index.nprobe = nprobe

    return index

def clustering_topk(Q, K, k, B, index, lsh_objects = None, verbose=False):
    n,d = Q.size()

    # Append a zero to each query vector to match the dimension of the key vectors.
    Q = torch.cat((Q, torch.zeros(Q.size(0), 1)), dim=1)

    # Search for each query vector in Q in the index for
    # its k nearest neighbors.
    # This takes O(nk) time.
    D, I = index.search(Q.numpy(), k)

    D = torch.tensor(D)
    print(D)

    # For each row of D, we have the distances ||q - k_i||.
    # We will use these to calculate the attention scores.
    # score = 0.5 * (||q||^2 + ||k_i||^2 - ||q - k_i||^2)
    # Note that ||k_i||^2 = d*B*B
    
    # First, we calculate ||q||^2 -> O(nd) time
    q_norm = torch.sum(Q**2, dim=1, keepdim=True)

    # Next, we calculate ||k_i||^2
    k_norm = K.size(1)*B*B

    # Finally, we calculate the attention scores
    attention_scores = 0.5 * (q_norm + k_norm - D**2)
    attention_scores = torch.exp(attention_scores / d - B*B)

    return attention_scores, I

@profile
def clustering_topk_preprocessing_ip_index(K, B, voronoi_cells=100, nprobe=10, verbose=False, Q=None, k=None):
    n,d = K.size()
    # We will use the Faiss library to perform the clustering.

    # Build a FAISS Flat index:
    index = faiss.IndexFlatIP(d)

    K_normalized = K.clone()

    # Add the normalized key vectors to the index
    index.add(K_normalized.numpy().astype('float32'))

    # Search for each query vector in Q in the index for
    # its k nearest neighbors.
    # This takes O(nk) time.
    D, I = index.search(Q.numpy().astype('float32'), k)
    D = torch.tensor(D) / d

    # For numerical stability, subtract the maximum value from each row of D
    max_D = torch.max(D, dim=1, keepdim=True)[0]
    D = D - max_D

    # Finally, we calculate the attention scores
    attention_scores = torch.exp(D)

    return attention_scores, I, max_D

def clustering_topk_ip_index(Q, K, k, B, index, lsh_objects = None, verbose=False):
    pass

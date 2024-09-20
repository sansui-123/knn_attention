import math
import numpy as np
import torch

# LSH class for approximate nearest neighbor search.
class angularLSH:
    def __init__ (self, K, s, b, B, verbose=False):
        self.K = K.clone()
        self.s = s
        self.b = b
        self.n, self.d = self.K.size()
        self.B = B

        # Now add an extra dimension to the key vectors to 
        # make them all have norm d*B*B - the maximum it could be. 
        # Now all the key vectors have d+1 dimensions.
        self.K = torch.cat((self.K, torch.sqrt(self.d*B*B - torch.sum(self.K**2, dim=1, keepdim=True))), dim=1)
        self.d += 1

        # LSH works as follows:
        # 1. To hash a single vector, concatenate k = O(log n) smaller hashes. 
        #    That's one hash table. That means we have 2^k = O(n) buckets
        # 2. Maintain L = n^s hash tables where s = log(1-arccos(b)/pi) / log(1-arccos(s)/pi)
        # 
        # HEY!
        #   Note we might need to adjust the number of hash tables and the number of bits in each hash.
        #   This is a heuristic.
        self.k = int(0.5 * math.ceil(math.log2(self.n) / (-math.log2(1-np.arccos(self.s / (self.d * B * B))/math.pi))))
        self.L = int(0.5 * math.log2(1-np.arccos(self.b / (self.d*B*B))/math.pi) / math.log2(1-np.arccos(self.s / (self.d * B *B))/math.pi))

        # if verbose:
        #     print(f"Power of L: {self.L}")

        self.L = int(math.pow(self.n, self.L))

        if verbose:
            print(f"Creating {self.L} hash tables with {self.k} bits each.")

        # A hash is just the sign of the dot product of the vector 
        # with a random vector on the unit sphere.
        # (Coordinates are drawn from a Gaussian distribution.)
        self.hash_vectors = torch.randn(self.L, self.k, self.d)

        # We'll store L hash tables, each with 2^k buckets. Each bucket
        # will store a list of indices of the key vectors.
        self.hash_tables = [{} for _ in range(self.L)]

        # Now we hash all the key vectors L times:
        for j in range(self.L):
            for i in range(self.n):
                h = self.hash(self.K[i], self.hash_vectors[j])
                # print(f"Hashing key {self.K[i]} to bucket {h} in hash table {j}")

                if h not in self.hash_tables[j]:
                    self.hash_tables[j][h] = []
                self.hash_tables[j][h].append(i)

    #
    # Hash a single vector x using the hash function h.
    # h consists of k random vectors on the unit sphere.
    #
    # Parameters:
    #   x is the input vector: 1 x (d+1)
    #   h is the hash function: k x (d+1)
    # 
    # Returns:
    #   The hash of the input vector x: a number between 0 and 2^k - 1
    # 
    def hash(self, x, h):
        
        assert x.size() == torch.Size([self.d]), f"Expected size {self.d}, got {x.size()}"

        hash_value = 0
        for i in range(self.k):
            # print(f"Dot product: {x} and {h[i]}")
            if torch.dot(x, h[i]) >= 0:
                hash_value += 2**i

        return hash_value

    #
    # Function to query how many (and which) key vectors are in the same bucket 
    # as the query vector q.
    # 
    # We limit the number of returned key vectors to max_results.
    # We also require the keys to have a dot product of at most cr with the query vector.
    #
    # Runtime: O(max(max_results, number of keys in the bucket))
    def query_bucket_size(self, q, max_results):

        assert q.size() == torch.Size([self.d]), f"Expected size {self.d}, got {q.size()}"

        distinct_keys = set()
        for j in range(self.L):
            hj = self.hash(q, self.hash_vectors[j])

            # print(f"Query in hash table {j} was hashed at bucket {hj}")

            if hj in self.hash_tables[j]:
                for key_index in self.hash_tables[j][hj]:

                    # Check if the key vector has a dot product of at most b with the query vector.
                    if torch.dot(q, self.K[key_index]) > self.b:
                        continue
                    
                    # Add the key index to the set of distinct keys.
                    distinct_keys.add(key_index)

                    # If we have reached the maximum number of results, return the keys.
                    if len(distinct_keys) >= max_results:
                        return distinct_keys, len(distinct_keys)
                    
        # print("Found", len(distinct_keys), "distinct keys.")

        return distinct_keys, len(distinct_keys)
    
    def print_buckets(self):
        print(f"Printing {self.L} hash tables:")
        for j in range(self.L):
            print("Hash table", j)
            for key in self.hash_tables[j]:
                print(key, ":", self.hash_tables[j][key])
            print()
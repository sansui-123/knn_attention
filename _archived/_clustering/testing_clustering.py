import torch
import time
import os
import numpy as np
from _clustering.clustering_topk import clustering_topk, clustering_topk_preprocessing_flatl2, clustering_topk_preprocessing_ivff_index
from _clustering.clustering_topk import clustering_topk_ip_index, clustering_topk_preprocessing_ip_index
from forward_pass import forward_pass_unoptimized
from naive_attention import naive_attention
import math

def run_experiment(Q, K, V, n, d, B, run_exact=True, store=False, verbose=False):

    # After a certain threshold, increasing the number of clusters does not improve the accuracy.
    k = int(math.pow(n, 1/2))
    l = 1

    print(f"k = {k}, l = {l}")

    # Call the clustering_topk_preprocessing function.
    # Measure the time it takes to run the function.
    start_time = time.time()
    topk_scores, topk_indices, normalizer = clustering_topk_preprocessing_ip_index(K, B, voronoi_cells=100, nprobe=10, verbose=False, Q=Q, k=k)

    # Get the top-k attention scores and indices.
    # topk_scores, topk_indices = clustering_topk(Q, K, k, B, index, verbose=False)

    print("Time taken for clustering_topk_preprocessing:", time.time() - start_time)

    # Call the sampling_attention function.
    attention_output = \
        forward_pass_unoptimized.sampling_attention_clustering_v2(Q, K, V, \
                                                            k, l, \
                                                            topk_scores, \
                                                            topk_indices, \
                                                            normalizer)
    end_time = time.time()
    print("Time taken for approximate attention:", end_time - start_time)

    if verbose:
        print(attention_output)
        print("\n\n")

    if not run_exact:
        exit()

    # Compare with the exact attention output

    # If we have saved the result to a file, we can load it here.
    output_path = "exact_attention_output_" + str(n) + "_" + str(d) + ".pt"
    time_path = "exact_attention_time_" + str(n) + "_" + str(d) + ".txt"

    if os.path.exists(output_path) and store:
        print("Loading exact attention output from file.")
        exact_attention_output = torch.load(output_path)

        # We should also have the time taken to calculate the exact attention.
        with open(time_path, "r") as f:
            exact_time = float(f.readline())
        
        print("Time taken for exact attention:", exact_time)
        max_error = torch.max(torch.abs(attention_output - exact_attention_output))
        print("Max error: ", torch.max(torch.abs(attention_output - exact_attention_output)))
        print("Mean error: ", torch.mean(torch.abs(attention_output - exact_attention_output)))

        if verbose:
            print(exact_attention_output)
    else:
        print("Calculating exact attention.")
        # Calculate the time taken to run the calculate_attention function.
        start_time = time.time()
        exact_attention_output = naive_attention.calculate_attention(Q, K, V)
        end_time = time.time()
        print("Time taken for exact attention:", end_time - start_time)

        if verbose:
            print(exact_attention_output)

        # Print the mean absolute error
        max_error = torch.max(torch.abs(attention_output - exact_attention_output))
        print("Max error: ", max_error)
        print("Mean error: ", torch.mean(torch.abs(attention_output - exact_attention_output)))

        if store:
            # Save result to file.
            # The file should be named: exact_attention_output_n_d.pt
            torch.save(exact_attention_output, output_path)

            # Write the time taken to a file also. The file should be named: exact_attention_time_n_d.txt
            with open(time_path, "w") as f:
                f.write(str(end_time - start_time))

    return max_error

torch.manual_seed(0)
n, d = 10000, 700
B = 100

# Generate n x d random matrices with entries in [-B,B].
# These are real numbers.
Q = torch.rand(n, d) * 2 * B - B
K = torch.rand(n, d) * 2 * B - B
V = torch.rand(n, d) * 2 * B - B

# Cast to float32
Q = Q.float()
K = K.float()
V = V.float()

num_iterations = 1

estimate = 0
for i in range(num_iterations):
    estimate += run_experiment(Q, K, V, n, d, B, run_exact=True, store=True, verbose=False)

print("Average max error = ", estimate / num_iterations)
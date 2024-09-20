import torch
from softmax_expectation.softmax_expectation import softmax_sample, softmax_expectation_estimation
from softmax_expectation.topk import topk
import matplotlib.pyplot as plt

def experiment():
    # Input sequence length
    N = 10000
    d = 5

    mu = 0
    std = 10

    Q = torch.randn(N, d) * std + mu
    K = torch.randn(N, d) * std + mu

    # Sample from the softmax Q[0] @ K[j]^T for j in [N].
    # Do this 100 times and form a histogram of the indices.
    histogram_true = torch.zeros(N)
    distribution = torch.softmax(Q[0] @ K.T, dim=0)
    for _ in range(100):
        index = torch.multinomial(distribution, 1).item()
        histogram_true [index] += 1

    # Now we do the same thing but with our method.
    # But first, our method requires the top sqrt(N) indices of Q[0] @ K[j]^T.
    c=0.4
    scores, S = topk(Q, K, int((c*N) ** 0.5))
    scores = torch.tensor(scores, dtype=torch.float64)
    histogram_sampled = torch.zeros(N)
    for _ in range(100):
        index = softmax_sample(Q, K, 0, S[0,:].tolist(), scores[0,:])
        histogram_sampled[index] += 1

    # Draw the two histograms.
    # The two histograms should be similar. If they are not, there is a bug in the sampling method.
    # plt.bar(range(N), histogram_true)
    # plt.bar(range(N), histogram_sampled)
    # plt.show()

    # Now we will test the expectation method.
    # We will calculate the expectation of the function f(j) = j.
    # This is the expected value of the sampled index.
    # Analytically, we just have to evaluate the inner product of (Q[0] @ K[j]^T)_{j=1}^n * f.

    def f(Q, K, V, dO, index, ii, jj):
        return index * index * index

    expectation_true = torch.sum(distribution * torch.arange(N) ** 3)
    expectation_sampled = softmax_expectation_estimation(Q, K, 0, f, None, S[0,:].tolist(), scores[0,:], epsilon=0.01, delta=0.05)

    print(f"True expectation: {expectation_true}")
    print(f"Sampled expectation: {expectation_sampled}")
    print(f"Relative error: {torch.abs(expectation_true - expectation_sampled) / expectation_true}")

    return torch.abs(expectation_true - expectation_sampled) / expectation_true

mean_error = 0
num_experiments = 50
for _ in range(num_experiments):
    print(f"Experiment {_}")
    mean_error += experiment()

print(f"Mean relative error: {(mean_error / num_experiments) * 100}%")
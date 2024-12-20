import numpy as np

def select(A_samples, B_samples):
    selected_indices = []
    for a_index in range(A_samples.shape[0]):
        print(a_index)
        a_sample = A_samples[a_index]
        correlation_scores = []
        for b_index in range(B_samples.shape[0]):
            b_sample = B_samples[b_index]
            correlations = []
            for dim in range(a_sample.shape[0]):
                correlation = np.corrcoef(a_sample[dim], b_sample[dim])[0, 1]
                correlations.append(correlation)
            average_correlation = np.mean(correlations)
            correlation_scores.append((average_correlation, b_index))
        correlation_scores.sort(reverse=True, key=lambda x: x[0])
        top_10_indices = [index for score, index in correlation_scores[:10]]
        print(top_10_indices)
        selected_indices.extend(top_10_indices)
    selected_indices = list(set(selected_indices))

    selected_samples = B_samples[selected_indices]

    print("Number of reconstruction samples:", selected_samples.shape[0])

# select(target_DZ_combined_tensor,source_DZ_combined_tensor)

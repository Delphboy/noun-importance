from typing import Optional

import matplotlib.pyplot as plt


def print_top_k(weights: dict, k: Optional[int] = 10):
    sorted_weights = {
        k: v for k, v in sorted(weights.items(), key=lambda item: item[1], reverse=True)
    }
    for i, key in enumerate(sorted_weights):
        if i == k:
            break
        print(f"{i + 1}. {key}: {sorted_weights[key]}")


def get_minimum(weights: dict) -> tuple:
    min_weight = 100000
    min_noun = ""
    for weight in weights:
        w = weights[weight]
        if w < min_weight:
            min_noun = weight
            min_weight = w
    return min_noun, min_weight


def get_maximum(weights: dict) -> tuple:
    max_weight = 0
    max_noun = ""
    for weight in weights:
        w = weights[weight]
        if w > max_weight:
            max_noun = weight
            max_weight = w
    return max_noun, max_weight


def get_average(weights: dict) -> float:
    running_total = 0
    for weight in weights:
        running_total += weights[weight]
    return running_total / len(weights)


def plot_histogram(weights: dict, filename: Optional[str] = "histogram.png"):
    plt.figure(figsize=(10, 5))
    plt.hist(weights.values(), bins=50)
    plt.title("Histogram of Weights")
    plt.xlabel("Weight")
    plt.ylabel("Frequency")
    plt.savefig(filename)
    # plt.show()


def normalise_weights(weights: dict) -> dict:
    normalised_weights = {}
    total = 0
    for key in weights:
        total += weights[key]
    for key in weights:
        normalised_weights[key] = weights[key] / total
    return normalised_weights

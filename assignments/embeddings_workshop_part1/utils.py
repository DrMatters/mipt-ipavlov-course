import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import cosine as cosine_distance


def plot_moving_average(series, window, plot_intervals=False, plot_actual=True, scale=1.96, plot_anomalies=False):
    """
        series - dataframe with timeseries
        window - rolling window size
        plot_intervals - show confidence intervals
        plot_anomalies - show anomalies

    """
    rolling_mean = series.rolling(window=window).mean()

    plt.figure(figsize=(15, 5))
    plt.title("Moving average\n window size = {}".format(window))
    plt.plot(rolling_mean, "g", label="Rolling mean trend")

    # Plot confidence intervals for smoothed values
    if plot_intervals:
        mae = mean_absolute_error(series[window:], rolling_mean[window:])
        deviation = np.std(series[window:] - rolling_mean[window:])
        lower_bond = rolling_mean - (mae + scale * deviation)
        upper_bond = rolling_mean + (mae + scale * deviation)
        plt.plot(upper_bond, "r--", label="Upper Bond / Lower Bond")
        plt.plot(lower_bond, "r--")

        # Having the intervals, find abnormal values
        if plot_anomalies:
            anomalies = pd.DataFrame(index=series.index, columns=series.columns)
            anomalies[series < lower_bond] = series[series < lower_bond]
            anomalies[series > upper_bond] = series[series > upper_bond]
            plt.plot(anomalies, "ro", markersize=10)

    if plot_actual:
        plt.plot(series[window:], label="Actual values")
    plt.legend(loc="upper left")
    plt.grid(True)


class EmbeddingsEval:
    def __init__(self, matrix, n_neighbors=5, words_to_tokens=None, tokens_to_words=None):
        self.matrix = matrix
        self.words_to_tokens = words_to_tokens
        self.tokens_to_words = tokens_to_words
        self.neighbors = NearestNeighbors(n_neighbors=n_neighbors, metric=cosine_distance)
        self.neighbors.fit(matrix)

    def tokens_to_neighbors(self, tokens, n_neighbors=5):
        vectors = self.tokens_to_embeddings(tokens)
        return self.neighbors.kneighbors(vectors, n_neighbors + 1, return_distance=False)[1:]

    def words_to_neighbors(self, words, n_neighbors=5):
        vectors = self.words_to_embeddings(words)
        return self.neighbors.kneighbors(vectors, n_neighbors + 1, return_distance=False)[:, 1:]

    def words_to_embeddings(self, words):
        tokens = self.words_to_tokens(words)
        return self.matrix[tokens, :]

    def tokens_to_embeddings(self, tokens):
        return self.matrix[tokens, :]

    def most_similar(self, positive=[], negative=[], n_neighbors=5):
        avg = np.zeros((1, self.matrix.shape[1]))
        if len(positive):
            avg += np.mean(self.tokens_to_embeddings(self.words_to_tokens(positive)), axis=0)
        if len(negative):
            avg -= np.mean(self.tokens_to_embeddings(self.words_to_tokens(negative)), axis=0)
        similar = self.neighbors.kneighbors(avg, n_neighbors, return_distance=False)
        return self.tokens_to_words(similar[0])


def adjust_learning_rate(optimizer, factor=0.5):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    for param_group in optimizer.param_groups:
        old_lr = param_group['lr']

    for param_group in optimizer.param_groups:
        param_group['lr'] = old_lr * factor

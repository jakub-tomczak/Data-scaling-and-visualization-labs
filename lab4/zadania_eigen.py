#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg


def vectors_uniform(k):
    """Uniformly generates k vectors."""
    vectors = []
    for a in np.linspace(0, 2 * np.pi, k, endpoint=False):
        vectors.append(2 * np.array([np.sin(a), np.cos(a)]))
    return vectors


def visualize_transformation(A, vectors):
    """Plots original and transformed vectors for a given 2x2 transformation matrix A and a list of 2D vectors."""
    for i, v in enumerate(vectors):
        # Plot original vector.
        plt.quiver(0.0, 0.0, v[0], v[1], width=0.008, color="blue", scale_units='xy', angles='xy', scale=1,
                   zorder=4)
        plt.text(v[0] / 2 + 0.25, v[1] / 2, "v{0}".format(i), color="blue")

        # Plot transformed vector.
        tv = A.dot(v)
        plt.quiver(0.0, 0.0, tv[0], tv[1], width=0.005, color="magenta", scale_units='xy', angles='xy', scale=1,
                   zorder=4)
        plt.text(tv[0] / 2 + 0.25, tv[1] / 2, "v{0}'".format(i), color="magenta")
    plt.xlim([-6, 6])
    plt.ylim([-6, 6])
    plt.margins(0.05)
    # Plot eigenvectors
    plot_eigenvectors(A)
    plt.show()


def visualize_vectors(vectors, color="green"):
    """Plots all vectors in the list."""
    for i, v in enumerate(vectors):
        plt.quiver(0.0, 0.0, v[0], v[1], width=0.006, color=color, scale_units='xy', angles='xy', scale=1,
                   zorder=4)
        plt.text(v[0] / 2 + 0.25, v[1] / 2, "eigv{0}".format(i), color=color)


def plot_eigenvectors(A):
    """Plots all eigenvectors of the given 2x2 matrix A."""
    _, eigvec = linalg.eig(A)  # eig zwraca eigenvalues na pozycji 0, eigenvectors na pozycji 1
    visualize_vectors(eigvec.T)


def EVD_decomposition(A):
    eig_val, eig_vec = linalg.eig(A)
    L = np.eye(len(eig_val)) * eig_val
    K = eig_vec
    K_inverted = linalg.inv(K)
    assert np.allclose(K @ L @ K_inverted, A), 'EVD decomposition is invalid'
    return K, L, K_inverted

def normalize_vector(vec):
    return vec / linalg.norm(vec)

def find_unique_vectors(vectors):
    return np.unique(vectors, axis=0)


def plot_attractors(A, vectors):
    colors = np.array([['seagreen', 'steelblue'], ['orange', 'lightcoral'], ['darkorange', 'indianred'], ['forestgreen', 'lightskyblue']])
    eig_val, eig_vec = linalg.eig(A)
    unique_val, unique_vec = find_unique_vectors(eig_val), find_unique_vectors(eig_vec.T)
    inversed_val, inversed_vec = unique_val*(-1), unique_vec*(-1)

    def _plot_attractor(A, vectors, unique_vec, unique_inversed_vec):
        for i, v in enumerate(vectors):
            norm_vector = normalize_vector(v)
            temp_vec = norm_vector.copy()
            for _ in range(5):
                temp_vec = normalize_vector(A.dot(temp_vec))

            min_index = np.argmin(
                [np.sum(np.abs(temp_vec - eigen_vector))
                    for eigen_vector in np.concatenate((unique_inversed_vec, unique_vec))]
            )
            color = colors[min_index % len(unique_vec), min_index // len(unique_vec)]

            plt.quiver(0.0, 0.0, -norm_vector[0], -norm_vector[1], width=0.004, color=color, scale_units='xy', angles='xy', scale=1,
                       zorder=1)

    _plot_attractor(A, vectors, unique_vec, inversed_vec)
    for i, v in enumerate(unique_vec):
        color_index = min(i, len(colors) - 1)
        plt.quiver(0.0, 0.0, v[0], v[1], width=0.012, color=colors[color_index, 0], scale_units='xy', angles='xy', scale=1,
                   zorder=4)
        plt.text(v[0] + .1, v[1] + .1, "{}".format(np.round(eig_val[i], 2)), color=colors[color_index, 0])

        plt.quiver(0.0, 0.0, -v[0], -v[1], width=0.012, color=colors[color_index, 1], scale_units='xy', angles='xy', scale=1,
                   zorder=4)
        plt.text(-v[0] - .2, -v[1] - .2, "{}".format(np.round(eig_val[i], 2)), color=colors[color_index, 1])

    lim = 3
    plt.grid()
    plt.xlim([-lim, lim])
    plt.ylim([-lim, lim])
    plt.margins(0.05)
    plt.show()

def runner(func):
    def dec(*args, **kwargs):
        A, vectors = func(*args, **kwargs)
        visualize_transformation(A, vectors)
        K, L, K_inverted = EVD_decomposition(A)
        plot_attractors(A, vectors)

    return dec


@runner
def test_A1(vectors):
    """Standard scaling transformation."""
    A = np.array([[2, 0],
                  [0, 2]])
    A = np.array([[3, 1],
                  [0, 2]])
    return A, vectors


@runner
def test_A2(vectors):
    A = np.array([[-1, 2],
                  [2, 1]])
    return A, vectors


@runner
def test_A3(vectors):
    A = np.array([[3, 1],
                  [0, 2]])
    return A, vectors


if __name__ == "__main__":
    vectors = vectors_uniform(k=8)
    test_A1(vectors)
    test_A2(vectors)
    test_A3(vectors)

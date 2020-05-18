from sklearn.manifold import Isomap, MDS, TSNE, LocallyLinearEmbedding
from sklearn.decomposition import PCA
from sklearn.datasets import make_swiss_roll
import pandas as pd
from typing import Tuple, Any, List, Dict
import matplotlib.pyplot as plt
import numpy as np
from time import time
import os
from functools import partial
from scipy.spatial.distance import cdist


# for running outside this file's dir
BASE_PATH='lab5'

def centering_matrix(n):
    return np.identity(n) - 1 / n * np.ones(n)

class ManualPCA:
    def __init__(self, n_components=2):
        self.n_components = n_components

    def fit_transform(self, data):
        data -= np.mean(data)
        data_euclidean = cdist(data, data, 'euclidean')
        J = centering_matrix(data_euclidean.shape[0])
        B = -.5 * J @ (data_euclidean**2) @ J

        eig_val, eig_vec = np.linalg.eig(B) # eig_vec are already normalized
        sorted_eig_values_indices = np.argsort(eig_val)[::-1][:self.n_components]
        sorted_eig_values, K = eig_val[sorted_eig_values_indices], eig_vec[:, sorted_eig_values_indices]
        L = np.eye(len(sorted_eig_values))*sorted_eig_values

        Y = K[:, :self.n_components] @ (L[:, :self.n_components] ** .5)
        return np.real(Y)

def transform_data(data: np.ndarray, method: Any, method_name: str):
    print("Transforming data using {:<30}".format(method_name), end="\t\t")
    s = time()
    result = method.fit_transform(data)
    e = time()
    total_time = e-s
    print("{:.4f} s".format(total_time))
    return result, total_time

def draw_plot(method_data: Tuple[Any, Tuple[np.ndarray, float]], dataset_name: str,
        data_labels: List[str] = None, data_colors: List[Any] = None):
    all_methods_count = len(method_data)
    max_rows = 3
    max_cols = 3
    min_x, max_x = 0, 20
    min_y, max_y = 0, 20
    x_y_margin = .5
    assert max_cols*max_rows >= all_methods_count, f'too many methods for one plot, max is {max_cols*max_rows}'

    fig = plt.figure(figsize=(5*max_rows, 5*max_cols))
    for i, (method, (data, transformation_time)) in enumerate(method_data):
        # print(f'Plotting results for {method}')
        axis = plt.subplot(max_rows, max_cols, i+1)
        min_x, max_x = np.min(data[:, 0])-x_y_margin, np.max(data[:, 0])+x_y_margin
        min_y, max_y = np.min(data[:, 1])-x_y_margin, np.max(data[:, 1])+x_y_margin
        axis.set_xlim([min_x, max_x])
        axis.set_ylim([min_y, max_y])
        axis.set_title('{:<30}{:.4f} sek.'.format(method, transformation_time))
        axis.scatter(data[:, 0], data[:, 1], c=data_colors, cmap=plt.cm.Spectral)
        if data_labels is not None:
            for ann_index, annotation in enumerate(data_labels):
                axis.annotate(xy=(data[ann_index, 0], data[ann_index, 1]), s=annotation)
        
        plt.suptitle(dataset_name)
        plt.savefig(os.path.join(BASE_PATH, f'data/{dataset_name}.png'))

    # plt.show()

def load_data(filename, **pandas_kwargs):
    print(f'loading data {filename}')
    return pd.read_csv(os.path.join(BASE_PATH, filename), **pandas_kwargs)


def transform_data_using_methods(data, methods, dataset_name, data_labels=None, data_colors=None):
    transformed_data = [(method_name, transform_data(data, method, method_name)) for method_name, method in methods.items()]
    draw_plot(transformed_data, dataset_name, data_labels, data_colors)


if __name__ == "__main__":
    # # używanie eigen_solver='auto' może prowadzić do błędu: Error in determining null-space with ARPACK. Error message: 'Factor is exactly singular'
    # # dodany random_state może pomóc, lepiej nie używać dense z dużymi macierzami
    # # ale mimo wszystko czasami należy zmienić na dense ze względu na wyżej wymienioony błąd.
    LLE = partial(LocallyLinearEmbedding,  n_components=2, eigen_solver='dense', random_state=21, n_jobs=1)

    methods = {
        "MDS sklearn, SMACOF": MDS(n_components=2),
        "MDS classic, PCA": ManualPCA(n_components=2),
        # "MDS classic (sklearn), PCA": PCA(n_components=2),
        "Isomap (k=3)": Isomap(n_neighbors=3, n_components=2, n_jobs=1),
        "Isomap (k=5)": Isomap(n_neighbors=5, n_components=2, n_jobs=1),
        "TSNE": TSNE(n_components=2, n_jobs=1),
        "TSNE (lr=50)": TSNE(n_components=2, learning_rate=50, n_jobs=1),
        "LLE (k=3)": LLE(n_neighbors=3),
        "LLE (k=5)": LLE(n_neighbors=5),
        "LLE (k=5, method=ltsa)": LLE(n_neighbors=5, method='ltsa')
    }

    print('Processing cars ...')
    data_file = 'data/cars.csv'
    data = load_data(data_file, sep=',', header=None)
    names = [x.replace("'", "") for x in data.loc[:, 0].values] # pobranie nazw i usunięcie '
    # usunięcie kolumny z nazwami oraz pobranie danych jako numpy array
    data = data.drop(data.columns[0], axis=1).values
    transform_data_using_methods(data, methods, "cars", data_labels=names)
    
    # swiss roll
    print("Processing swiss roll")
    data, colors = make_swiss_roll(1600)
    transform_data_using_methods(data, methods, "Swiss roll", data_colors=colors)

    # zbiór Facebook_metrics
    print('Processing Facebook metrics ...')
    data_file = 'data/dataset_Facebook.csv'
    data = load_data(data_file, sep=';').drop('Type', axis=1).dropna(axis=0).values
    transform_data_using_methods(data, methods, "Facebook metrics")

'''
Wnioski:
    - dla różnych zbiorów różne metody dają najlepsze wyniki
        - dla cars:
        - dla swiss roll:
        - dla facebook:
    - czas działania:
        SMACOF działa wolno
        reszta podobny czas
    -
    
'''
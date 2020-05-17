from sklearn.manifold import Isomap, MDS, TSNE, LocallyLinearEmbedding
from sklearn.decomposition import PCA
from sklearn.datasets import make_swiss_roll
import pandas as pd
from typing import Tuple, Any, List, Dict
import matplotlib.pyplot as plt
import numpy as np
from time import time
from zadania_pca import pca_manual


class ManualPCA:
    def __init__(self, n_components=2):
        self.n_components = n_components

    def fit_transform(self, data):
        return pca_manual(data, self.n_components)

def transform_data(data: np.ndarray, method: Any, method_name: str):
    print("Transforming data using {:<20}".format(method_name), end="\t\t")
    s = time()
    result = method.fit_transform(data)
    e = time()
    print("{:.4f} s".format(e-s))
    return result

def draw_plot(method_data: Tuple[Any, np.ndarray], dataset_name: str, data_labels: List[str] = None):
    all_methods_count = len(method_data)
    max_rows = 3
    max_cols = 3
    min_x, max_x = 0, 20
    min_y, max_y = 0, 20
    x_y_margin = .5
    assert max_cols*max_rows >= all_methods_count, f'too many methods for one plot, max is {max_cols*max_rows}'

    fig = plt.figure(figsize=(5*max_rows, 5*max_cols))
    for i, (method, data) in enumerate(method_data):
        # print(f'Plotting results for {method}')
        axis = plt.subplot(max_rows, max_cols, i+1)
        min_x, max_x = np.min(data[:, 0])-x_y_margin, np.max(data[:, 0])+x_y_margin
        min_y, max_y = np.min(data[:, 1])-x_y_margin, np.max(data[:, 1])+x_y_margin
        axis.set_xlim([min_x, max_x])
        axis.set_ylim([min_y, max_y])
        axis.set_title(method)
        axis.scatter(data[:, 0], data[:, 1])
        if data_labels is not None:
            for ann_index, annotation in enumerate(data_labels):
                axis.annotate(xy=(data[ann_index, 0], data[ann_index, 1]), s=annotation)
        
        plt.suptitle(dataset_name)
        plt.savefig(f'data/{method}.png')

    plt.show()

def load_data(filename, **pandas_kwargs):
    print(f'loading data {filename}')
    return pd.read_csv(filename, **pandas_kwargs)


def transform_data_using_methods(data, methods, dataset_name, data_labels=None):
    transformed_data = [(method_name, transform_data(data, method, method_name)) for method_name, method in methods.items()]
    draw_plot(transformed_data, dataset_name, data_labels)


if __name__ == "__main__":
    methods = {
        "MDS sklearn, SMACOF": MDS(n_components=2),
        "MDS manual, PCA": ManualPCA(n_components=2),
        "MDS classic, PCA": PCA(n_components=2),
        "Isomap (k=3)": Isomap(n_neighbors=3, n_components=2, n_jobs=4),
        "Isomap (k=5)": Isomap(n_neighbors=5, n_components=2, n_jobs=4),
        "TSNE": TSNE(n_components=2, n_jobs=4),
        "LLE (k=3)": LocallyLinearEmbedding(n_neighbors=3, n_components=2, n_jobs=4)
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
    data, _ = make_swiss_roll(100)
    transform_data_using_methods(data, methods, "Swiss roll")

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
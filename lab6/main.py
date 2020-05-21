import argparse
import numpy as np
from sklearn.decomposition import TruncatedSVD
from typing import Any, Dict
import skimage.io as io
import matplotlib.pyplot as plt
import os
from abc import abstractclassmethod


BASE_PATH=''
IMAGES_PATH=''

class SVDTransformer:
    def __init__(self, n_components: int):
        self.n_components = n_components

    def transform(self, image: np.ndarray):
        print(f'Metoda: {self}\nParametry:\n\tn_components {self.n_components}\n\twymiar {image.shape}')
        if self.n_components < 1:
            return image

        if len(image.shape) > 2:
            return np.stack(
                [self._transform(image[:, :, i]) for i in range(len(image.shape))],
                axis=-1)
        else:
            return self._transform(image)
    
    @abstractclassmethod
    def _transform(self, image: np.ndarray):
        '''
        Transforms one layer of an image.
        '''
        pass

class CustomSVDTransformer(SVDTransformer):
    def __init__(self, n_components):
        super().__init__(n_components)

    def _evd_decomposition(self, data):
        eig_val, eig_vec = np.linalg.eigh(data)
        sorted_eig_values_indices = np.argsort(eig_val)[::-1]
        L = eig_val[sorted_eig_values_indices]
        K = eig_vec[:, sorted_eig_values_indices]
        # K_inverted = np.real(np.linalg.inv(K))
        return K, L, None

    def _svd_decomposition(self, data):
        C = data.T @ data # columns covariance (m, m)
        R = data @ data.T # rows covariance (n, n)
        
        U, L_u, _ = self._evd_decomposition(R)
        V, L_v, _ = self._evd_decomposition(C)

        # sigma jest wektorem wartości własnych - tym który był dłuższy z C lub R
        sigma = np.power(L_v if C.shape[0] > R.shape[0] else L_u, .5)
        
        return U, sigma, V.T

    def _transform(self, data):
        n, m = data.shape
        k = self.n_components
        u, s, v_t = self._svd_decomposition(data)

        u_ = u[:, :k]
        v_t_ = v_t[:k, :]

        # dostosuj rozmiar sigmy do rozmiarów u_ oraz v_t_        
        s_ = np.diag(s)[:u_.shape[1], :v_t_.shape[0]]
        
        # print(u_[:10, :4], s_[:10, :4], v_t_[:10, :4], sep='\n')
        
        return np.real(u_ @ s_ @v_t_)

    def __repr__(self):
        return 'custom svd'

class ScikitTransformer(SVDTransformer):
    def __init__(self, n_components):
        super().__init__(n_components)
        self.method = TruncatedSVD(
            n_components=self.n_components
        )

    def _transform(self, data):
        k = self.n_components
        r, c = data.shape
        u, s, v_t = np.linalg.svd(data)
        
        # weź k wektorów własnych
        u_ = u[:, :k]
        v_t_ = v_t[:k, :]
        
        # stwórz macierz sigma tak by odpowiadała rozmiarom macierzy u_ oraz v_t_
        
        s_ = np.zeros((u_.shape[1], v_t_.shape[0]))
        s_len = s[:k].shape[0]
        s_[:s_len, :s_len] = np.diag(s[:k])
        
        # print(u_[:10, :4], s_[:10, :4], v_t_[:10, :4], sep='\n')
        return u_ @ s_ @ v_t_
        
    def __repr__(self):
        return 'scikit svd'

def transform_image(image: np.ndarray, method_obj: Any):
    '''
    Transform image using method_obj.
    method_obj should be an object that inherits from SVDTransformer.
    Returns transformed image.
    '''
    return method_obj.transform(image)

def read_image(image_filename: str, to_float=True):
    fullpath = os.path.join(BASE_PATH, IMAGES_PATH, image_filename)
    if not os.path.exists(fullpath):
        print(f"Nie odnaleziono pliku {fullpath}")
        exit(1)
    print(f'Czytanie pliku z {fullpath}')
    data = io.imread(fullpath)
    # convert from 0-255 to 0.0-1.0
    if to_float:
        data = data.astype('float64')
        if np.max(data) > 1.0:
            data = data / 255.0
    return data


def save_image(data: np.ndarray, image_filename: str):
    fullpath = os.path.join(BASE_PATH, image_filename)
    print(f'Zapisywanie pliku do {fullpath}')
    data = (data * 255).astype('uint8')
    io.imsave(fullpath, data)

def display_image(data: np.ndarray):
    io.imshow(data)
    plt.show()

def compare_images(orig_data: np.ndarray, transformed_data: np.ndarray, n_components: int, method: SVDTransformer):
    fig = plt.figure(figsize=(16, 8))
    plt.suptitle(f'Metoda {method}, liczba składowych: {n_components}')

    axis = plt.subplot(1, 2, 1)
    axis.set_title('Oryginalne zdjęcie')
    axis.imshow(orig_data)

    axis = plt.subplot(1, 2, 2)
    axis.set_title('Zdjęcie po kompresji')
    axis.imshow(transformed_data)
    plt.show()

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-f', type=str, dest="input_filename", required=True, help='plik z oryginalnym obrazkiem')
    parser.add_argument('-out', type=str, default=None, dest="output_filename",
        help='nazwa pliku wyjściowego, do którego zapisany ma być skompresowany obrazek. Brak wartości spowoduje wyświetlenie.')
    parser.add_argument('-svd', type=str, default='custom', dest="method_name",
        help='implementacja SVD do użycia. Możliwe wartości: `custom`(domyślna), `scikit`')
    parser.add_argument('-k', type=int, default=-1, dest="n_components",
        help='liczba wartości osobliwych użyta do kompresji (domyślnie wszystkie, czyli brak kompresji)')

    args = parser.parse_args()
    print(f'Podane argumenty: {args}')
    return args


def main(args):
    image = read_image(args.input_filename, to_float=True)
    
    # image = image[: 5,: 4, :]
    min_dim = np.min(image.shape[:2])
    if args.n_components > min_dim:
        print('n_components > min(m,n), zmiana n_components na', min_dim)
        args.n_components = min_dim


    # keeps methods and their constructor's parameters
    methods = {
        'custom': (CustomSVDTransformer, {
            'n_components': args.n_components
        }),
        'scikit': (ScikitTransformer, {
            'n_components': args.n_components
        })
    }
    if args.method_name not in methods.keys():
        print(f"Nieznana nazwa metody {args.method_name}")
        exit(1)

    # method initialization
    method_class, method_params = methods[args.method_name]
    method = method_class(**method_params)

    min_, max_ = np.min(image), np.max(image)
    # SVD decomposition
    transformed = np.clip(transform_image(image, method), min_, max_)

    if args.output_filename is not None:
        save_image(transformed, args.output_filename)
    else:
        compare_images(image, transformed, args.n_components, method)
        # display_image(transformed)

def test_custom_svd():
    m = np.array([
        [1, 2, 3, 4],
        [5, 6, 7, 8]
    ])

    m = np.random.rand(8, 6)

    # m = np.array([
    #     [1, 0, 0, 0, 2],
    #     [0, 0, 3, 0, 0],
    #     [0, 0, 0, 0, 0],
    #     [0, 4, 0, 0, 0]
    # ])


    custom_method = CustomSVDTransformer(n_components=np.min(m.shape)+1)
    scikit_method = ScikitTransformer(n_components=np.min(m.shape)+1)
    
    transformed_custom = transform_image(m, custom_method)
    transformed_scikit = transform_image(m, scikit_method)

    print('custom\n', transformed_custom)
    print('sickit\n', transformed_scikit)

    # check different sizes    
    m = m.T
    transformed_custom = transform_image(m, custom_method)
    transformed_scikit = transform_image(m, scikit_method)


if __name__=="__main__":
    # test_custom_svd()
    args = parse_args()
    main(args)
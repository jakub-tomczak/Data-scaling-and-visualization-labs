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
        print(f'Metoda: {self}\nParametry:\n\tn_components {self.n_components}')
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
        eig_val, eig_vec = np.linalg.eig(data)
        sorted_eig_values_indices = np.argsort(np.abs(eig_val))[::-1]
        # L = np.eye(len(eig_val)) * eig_val
        L = eig_val[sorted_eig_values_indices]
        K = eig_vec[sorted_eig_values_indices]
        K_inverted = np.real(np.linalg.inv(K))
        return np.real(K_inverted), np.real(L), np.real(K)

    def _svd_decomposition(self, data):
        C = data.T @ data # columns covariance
        R = data @ data.T # rows covariance
        # print('C shape is ', C.shape, 'R shape is', R.shape)
        V, L_v, V_t = self._evd_decomposition(C)
        U, L_u, U_t = self._evd_decomposition(R)


        # C_R_difference = np.abs(np.subtract(*data.shape))
        n, m = data.shape
        # print(n, m, C.shape, L_v.shape, L_u.shape)
        sigma = (L_v[:n] if m < n else L_u[:m]) ** .5
        # print(U.shape, sigma.shape, V.shape)
        # sigma_offset_fill = np.zeros((C_R_difference, sigma.shape[1]))
        # print(sigma.shape, sigma_offset_fill.shape)
        # sigma = np.vstack([sigma, sigma_offset_fill])

        return U, sigma, V

    def _transform(self, data):
        k = self.n_components
        # print(np.min(data), np.max(data))
        u, s, vh = self._svd_decomposition(data)
        # print(u.shape, s.shape, vh.shape)
        u_ = u[:, :k]
        s_ = np.eye(k)*s[:k]
        vh_ = vh[:k, :]
        # print(u_.shape, s_.shape, vh_.shape)
        transformed = (u_ @ s_ @vh_)
        # print(np.min(transformed), np.max(transformed))
        return np.clip(transformed, 0, 1)

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
        # print(np.min(data), np.max(data))
        u, s, vh = np.linalg.svd(data)
        # print(u.shape, s.shape, vh.shape)
        u_ = u[:, :k]
        s_ = np.eye(k)*s[:k]
        vh_ = vh[:k, :]
        # print(u_.shape, s_.shape, vh_.shape)
        transformed = (u_ @ s_ @vh_)
        # print(np.min(transformed), np.max(transformed))
        return np.clip(transformed, 0, 1)
        
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

    # SVD decomposition
    transformed = transform_image(image, method)

    if args.output_filename is not None:
        save_image(transformed, args.output_filename)
    else:
        compare_images(image, transformed, args.n_components, method)
        # display_image(transformed)

if __name__=="__main__":
    args = parse_args()
    main(args)
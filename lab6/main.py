import argparse
import numpy as np
from sklearn.decomposition import TruncatedSVD
from typing import Any, Dict
import skimage.io as io
import matplotlib.pyplot as plt
import os
from abc import abstractclassmethod


BASE_PATH='lab6'
IMAGES_PATH=''

class SVDTransformer:
    def __init__(self, n_components: int):
        self.n_components = n_components

    def transform(self, image: np.ndarray):
        print(f'Metoda: {self}\nParametry:\n\tn_components {self.n_components}')
        return self._transform(image)
    
    @abstractclassmethod
    def _transform(self, image: np.ndarray):
        pass

class CustomSVDTransformer(SVDTransformer):
    def __init__(self, n_components):
        super().__init__(n_components)
    
    def _transform(self, image):
        return image

    def __repr__(self):
        return 'custom svd'

class ScikitTransformer(SVDTransformer):
    def __init__(self, n_components):
        super().__init__(n_components)
        self.method = TruncatedSVD(
            n_components=self.n_components
        )
    
    def _transform(self, image):
        # _ = self.method.transform(image) # expects that image is an array with shapes (n_samples, n_features)
        return image
        
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

    image = read_image(args.input_filename, to_float=True)

    transformed = transform_image(image, method)

    if args.output_filename is not None:
        save_image(transformed, args.output_filename)
    else:
        display_image(transformed)

if __name__=="__main__":
    args = parse_args()
    main(args)
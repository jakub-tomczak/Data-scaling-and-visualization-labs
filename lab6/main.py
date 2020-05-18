import argparse

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-f', type=str, help='plik z oryginalnym obrazkiem')
    parser.add_argument('-out', type=str, default=None,
        help='nazwa pliku wyjściowego, do którego zapisany ma być skompresowany obrazek. Brak wartości spowoduje wyświetlenie.')
    parser.add_argument('-svd', type=str, default='custom', help='implementacja SVD do użycia. Możliwe wartości: `custom`(domyślna), `scikit`')
    parser.add_argument('-k', type=int, default=-1, help='liczba wartości osobliwych użyta do kompresji (domyślnie wszystkie, czyli brak kompresji)')

    args = parser.parse_args()
    print(f'Podane argumenty: {args}')
    return args

def main(args):
    pass

if __name__=="__main__":
    args = parse_args()
    main(args)
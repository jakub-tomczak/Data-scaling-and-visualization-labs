#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # Without this projection='3d' is not recognized


def combine_points(value_1, value_2, coeff):
    return coeff*value_1+(1-coeff)*value_2

def convex_combination(point_1, point_2, coeff):
    _x = combine_points(point_1[0], point_2[0], coeff)
    _y = combine_points(point_1[1], point_2[1], coeff)
    return np.array([_x, _y])

def draw_contour_2d(points):
    """Draws contour of the 2D figure based on the order of the points.

    :param points: list of numpy arrays describing nodes of the figure.
    """
    xs, ys = zip(points[-1], *points)
    plt.plot(xs, ys, color="blue")


def draw_contour_3d(points, sides):
    """Draws contour of the 3D figure based on the description of its sides.

    :param points: list of numpy arrays describing nodes of the figure.
    :param sides: list containing description of the figure's sides. Each side is described by a list of indexes of elements in points.
    """
    for side in sides:
        xs, ys, zs = [], [], []
        for s in side:
            xs.append(points[s][0])
            ys.append(points[s][1])
            zs.append(points[s][2])
        # Adding connection to the first node
        a = points[side[0]]
        xs.append(a[0])
        ys.append(a[1])
        zs.append(a[2])
        plt.plot(xs, ys, zs, color="blue")


def convex_comb_general(points, limit=1.0, step_arange=0.1, tabs=""):
    """Generates all linear convex combinations of points with the specified precision.

    :param points: list of numpy arrays representing nodes of the figure.
    :param limit: value to be distributed among remaining unassigned linear coefficients.
    :param step_arange: step in arange.
    :param tabs: indent for debug printing.
    :return: list of points, each represented as np.array.
    """
    # TODO: Zadanie 4.2: Rekurencyjna implementacja (albo zupełnie własna, rekurencja to propozycja).
    points = np.array(points)
    
    num_of_points = 10
    def add_at_position(elements, value, position):
        if position == 0:
            return (value, *elements)
        elif position == len(elements):
            return (*elements, value)
        return (*elements[:position], value, *elements[position:])

    def generate_coefficients(elements, number_of_elements):
        fill = np.arange(0, limit+1e-10, step=step_arange) # add small value to limit to include limit in the range
        
        if len(elements) == 1:
            return np.reshape(fill, (len(fill), 1))
        
        # get all permutations of the list smaller by 1 element
        permutations = generate_coefficients(elements[1:], number_of_elements) 
        
        tmp = []
        for permutation in permutations:
            if len(permutation) == number_of_elements-1:
                # add last element by subtracting sum of existing elements in the permutation from 1.0
                tmp.append(add_at_position(permutation, 1.0-np.sum(permutation), len(permutation)))
            else:
                for element in fill:
                    # add each possible value at the end if the sum is not bigger than 1.0
                    if np.sum(permutation) + element <= 1.0+1e-3:
                        permu = add_at_position(permutation, element, len(permutation))
                        tmp.append(permu) 

        return tmp

    
    # get all coefficients for convex combinations
    lambdas = np.array(generate_coefficients(np.zeros((len(points), )), len(points)))

    result = np.matmul(points.T, lambdas.T)

    return result


def convex_comb_triangle_loop(points):
    """Generates all linear convex combinations of points for a triangle using loops.

    :param points: list of numpy arrays representing nodes of the figure.
    :return: list of points, each represented as np.array.
    """
    assert len(points) == 3
    # TODO: Zadanie 4.1: Implementacja za pomocą pętli obliczenia wypukłych kombinacji liniowych dla wierzchołków trójkąta.
    num_of_points = 10
    coefficients = np.linspace(0, 1, num_of_points)
    
    _points = np.zeros(shape=(num_of_points**2, 2))
    for i, x in enumerate(coefficients):
        comb_point = convex_combination(points[1], points[2], x)
        for j, y in enumerate(coefficients):
            _points[i*num_of_points+j] = convex_combination(points[0], comb_point, y)

    return [x[0] for x in _points], [x[1] for x in _points]


def draw_convex_combination_2d(points, cc_points):
    # TODO: Zadanie 4.1: Rysowanie wykresu dla wygenerowanej listy punktów (cc_points).

    # Drawing contour of the figure (with plt.plot).
    plt.plot(cc_points[0], cc_points[1], 'ro')
    draw_contour_2d(points)


def draw_convex_combination_3d(points, cc_points, sides=None, color_z=True):
    fig = plt.figure()
    # To create a 3D plot a sublot with projection='3d' must be created.
    # For this to work required is the import of Axes3D: "from mpl_toolkits.mplot3d import Axes3D"
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    # TODO: Zadanie 4.3: Zaimplementuj rysowanie wykresu 3D dla cc_points. Możesz dodatkowo zaimplementować kolorowanie względem wartości na osi z.

    # Drawing contour of the figure (with plt.plot).
    ax.scatter(cc_points[0], cc_points[1], cc_points[2], c=cc_points[2])
    # ax.plot3D(cc_points[0], cc_points[1], cc_points[2], 'ro')
    if sides is not None:
        draw_contour_3d(points, sides)


def draw_vector_addition(vectors, coeffs):
    start = np.array([0.0, 0.0])
    for c, v in zip(coeffs, vectors):
        assert isinstance(v, np.ndarray)
        assert isinstance(c, float)
        # TODO: Zadanie 4.4: Wzorując się na poniższym użyciu funkcji plt.arrow, napisz kod rysujący wektory składowe.
        # TODO: Każdy kolejny wektor powininen być rysowany od punktu, w którym zakończył się poprzedni wektor.
        # TODO: Pamiętaj o przeskalowaniu wektorów przez odpowiedni współczynnik.

    # Drawing the final vector being a linear combination of the given vectors.
    # The third and the fourth arguments of the plt.arrow function indicate movement (dx, dy), not the ending point.
    resultant_vector = sum([c * v for c, v in zip(coeffs, vectors)])
    plt.arrow(0.0, 0.0, resultant_vector[0], resultant_vector[1], head_width=0.1, head_length=0.1, color="magenta", zorder=4, length_includes_head=True)
    plt.margins(0.05)



def draw_triangle_simple_1():
    points = [np.array([-1, 4]),
              np.array([2, 0]),
              np.array([0, 0])]
    cc_points = convex_comb_triangle_loop(points)
    draw_convex_combination_2d(points, cc_points)
    plt.show()

def draw_triangle_simple_2():
    points = [np.array([-2, 3]),
              np.array([4, 4]),
              np.array([3, 2])]
    cc_points = convex_comb_triangle_loop(points)
    draw_convex_combination_2d(points, cc_points)
    plt.show()

def draw_triangle_1():
    points = [np.array([-1, 4]),
              np.array([2, 0]),
              np.array([0, 0])]
    cc_points = convex_comb_general(points)
    draw_convex_combination_2d(points, cc_points)
    plt.show()

def draw_triangle_2():
    points = [np.array([-2, 3]),
              np.array([4, 4]),
              np.array([3, 2])]
    cc_points = convex_comb_general(points)
    draw_convex_combination_2d(points, cc_points)
    plt.show()

def draw_rectangle():
    points = [np.array([0, 0]),
              np.array([0, 1]),
              np.array([1, 1]),
              np.array([1, 0])]
    cc_points = convex_comb_general(points)
    draw_convex_combination_2d(points, cc_points)
    plt.show()

def draw_hexagon():
    points = [np.array([1, -2]),
              np.array([-1, -2]),
              np.array([-2, 0]),
              np.array([-1, 2]),
              np.array([1, 2]),
              np.array([2, 0])]
    cc_points = convex_comb_general(points)
    draw_convex_combination_2d(points, cc_points)
    plt.show()

def draw_not_convex():
    points = [np.array([0, 0]),
              np.array([0, 2]),
              np.array([1, 1]),
              np.array([2, 3]),
              np.array([2, 0])]
    cc_points = convex_comb_general(points)
    draw_convex_combination_2d(points, cc_points)
    plt.show()

def draw_tetrahedron():
    sides = [[0,1,2], [1,2,3], [0,2,3], [0,1,3]]
    points = [np.array([1.0, 1.0, 1.0]),
              np.array([-1.0, -1.0, 1.0]),
              np.array([-1.0, 1.0, -1.0]),
              np.array([1.0, -1.0, -1.0])]
    cc_points = convex_comb_general(points, step_arange=0.1)
    draw_convex_combination_3d(points, cc_points, sides=sides, color_z=True)
    plt.show()

def draw_cube():
    sides = [[0,1,2,3], [4,5,6,7], [0,4,5,1], [2,6,7,3]]
    points = [np.array([0.0, 0.0, 0.0]),
              np.array([1.0, 0.0, 0.0]),
              np.array([1.0, 1.0, 0.0]),
              np.array([0.0, 1.0, 0.0]),
              np.array([0.0, 0.0, 1.0]),
              np.array([1.0, 0.0, 1.0]),
              np.array([1.0, 1.0, 1.0]),
              np.array([0.0, 1.0, 1.0])]
    cc_points = convex_comb_general(points, step_arange=0.2)
    draw_convex_combination_3d(points, cc_points, sides=sides, color_z=True)
    plt.show()

def draw_vector_addition_ex1():
    v = [np.array([-1, 4]),
         np.array([2, 0]),
         np.array([0, 0])]
    coeffs = [0.4, 0.3, 0.3]
    draw_convex_combination_2d(v, convex_comb_general(v))
    draw_vector_addition(v, coeffs)
    plt.show()
    coeffs = [0.2, 0.8, 0.0]
    draw_convex_combination_2d(v, convex_comb_general(v))
    draw_vector_addition(v, coeffs)
    plt.show()



if __name__ == "__main__":
    # for task 4.1
    # draw_triangle_simple_1()
    # draw_triangle_simple_2()

    # for task 4.2
    draw_triangle_1()
    draw_triangle_2()
    draw_rectangle()
    draw_hexagon()
    # draw_not_convex()

    # for task 4.3
    # draw_tetrahedron()
    # draw_cube()

    # for task 4.4
    # draw_vector_addition_ex1()
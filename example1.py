import numpy as np
from matplotlib import pyplot as pl
from pycomponents import component

NUMBER_OF_SAMPLES = 100
PROPORTIONALITY_CONSTANT = 3
xvalues = np.linspace(0, 1, NUMBER_OF_SAMPLES)
yvalues = [np.random.normal(0.5, 0.15, 1)[0] - 0.25 * x for x in xvalues]
example_array = np.array([xvalues, yvalues])
principal_components = list(component.ComponentEvaluator(example_array.T))

# Proportionality adjustment
principal_components[0].length = principal_components[0].length * PROPORTIONALITY_CONSTANT
principal_components[1].length = principal_components[1].length * PROPORTIONALITY_CONSTANT

def plot_lin_eq(data, components):
    p1, p2 = get_linear_edge(components[0])
    p3, p4 = get_linear_edge(components[1])
    figure, axes = pl.subplots()
    axes.set_xlabel("$X_1$")
    axes.set_ylabel("$X_2$")
    axes.set_aspect('equal')
    axes.plot([p1.x, p2.x], [p1.y, p2.y], "--", color="blue")
    axes.plot([p3.x, p4.x], [p3.y, p4.y], "--", color="blue")
    axes.plot(data[0], data[1], ".", color="gray")
    return figure, axes



def get_linear_edge(component):
    theta = np.arctan(component.v[1] / component.v[0])
    delta_y = component.length * np.sin(theta)
    delta_x = component.length * np.cos(theta)
    max_x = component.center[0] + (delta_x / 2.)
    min_x = component.center[0] - (delta_x / 2.)
    max_y = component.center[1] + (delta_y / 2.)
    min_y = component.center[1] - (delta_y / 2.)
    return Point(min_x, min_y), Point(max_x, max_y)


class Point(object):

    def __init__(self, x, y):
        self.x = x
        self.y = y


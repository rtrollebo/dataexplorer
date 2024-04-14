"""
nnet architecture
"""
import numpy as np


class Layer:
    def __init__(self, size: int, name=None):
        self.size = size
        self.name = name if name is not None else ""

    def __repr__(self):
        return "Layer "+self.name


class Architecture:

    def __init__(self, layers: [Layer], activation_function=None):
        """
        A type for the architecture of the nnet
        :param layers: The layers of the architecture
        :param activation_function: The activation function of the architecture. (if not provided, a sigmoid function
        is used)
        """
        self.layers = layers
        self.num_layers = len(self.layers)
        if activation_function is None:
            self.activation_function = lambda v: 1 / (1 + np.exp(-v))


class Network:

    def __init__(self, architecture: Architecture, nodes: [np.ndarray], nodes_target: np.ndarray,
                 weights: [np.ndarray], error=None, learning_rate=1):
        """
        A type for the data of the nnet
        :param architecture: The architecture of the network
        :param nodes: The nodes of the network
        :param nodes_target: The target of the network
        :param weights: The weights of the network
        :param learning_rate: The learning rate of the network
        :param error: The error of an iteration of the network (a dynamic parameter used when iterations are run)
        """
        if error is None:
            error = [None, None, None]
        self.architecture = architecture
        self.nodes = nodes
        self.nodes_target = nodes_target
        self.weights = weights
        self.error = error
        self.learning_rate = learning_rate
        self.validate()

    def initiate(self):
        """
        Initial weights of the nnet
        :return:
        """
        for i, layer in enumerate(self.architecture.layers):
            if i == 0:
                continue
            self.weights[i] = self.weights[i] = np.random.uniform(
                -1,
                1,
                (self.architecture.layers[i-1].size, self.architecture.layers[i].size))

    def validate(self):
        self._validate_network()
        self._validate_nodes()
        self._validate_nodes_target()
        self._validate_weights()

    def _validate_network(self):
        if self.nodes is None or len(self.nodes) == 0:
            raise ValueError("Network must contain nodes")
        if self.weights is None or len(self.weights) == 0:
            raise ValueError("Network must contain weights")
        if self.nodes_target is None or len(self.nodes_target) == 0:
            raise ValueError("Network must contain target nodes")

    def _validate_nodes(self):
        for i, n in enumerate(self.nodes):
            if n is None or n.size != self.architecture.layers[i].size:
                raise ValueError("Empty nodes in layer={0}".format(i))
            else:
                if n.size != self.architecture.layers[i].size:
                    raise ValueError("node and architecture layer is not of equal size (layer={0})".format(n))

    def _validate_nodes_target(self):
        if self.architecture.layers[-1].size != self.nodes_target.size:
            raise ValueError("Target size must be equal to size of output layer")

    def _validate_weights(self):
        for i, w in enumerate(self.weights):
            if i == 0:
                continue
            if w is None:
                raise ValueError("Empty weights in layer={0}".format(i))
            else:
                if w.size != self.nodes[i].size*self.nodes[i-1].size:
                    raise ValueError("node- and weight size are not consistent in layer = {0}".format(i))



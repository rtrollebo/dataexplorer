"""
nnet operations module
"""
import doctest
import numpy as np
try:
    from nnet.architecture import Architecture
    from nnet.architecture import Network, Layer
except ImportError:
    # For running temporary doctests
    from architecture import Architecture
    from architecture import Network, Layer


def forward_pass(network: Network):
    """
    Conducts the forward pass of a network model based on input data and a network architecture
    :param network: The network
    :return:

    >>> nodes1 = [np.array([0.35, 0.9]), np.array([0.0, 0.0]), np.array([0.0])]
    >>> weights1 = [None, np.array([[0.1, 0.4], [0.8, 0.6]]), np.array([0.3, 0.9])]
    >>> network1 = Network(arc1, nodes1, np.array([0.5]), weights1, None)
    >>> forward_pass(network1)
    >>> network1.nodes[-1]
    0.6902834929076443
    >>>
    """
    for i, _ in enumerate(network.weights):
        if i == 0:
            continue
        network.nodes[i] = network.architecture.activation_function(np.dot(network.nodes[i-1], network.weights[i]))
    network.validate()


def calculate_delta(network: Network, layer: int):
    """
    Calculates the delta
    :param network: The network
    :param layer: The layer to calculates the delta at.
    :return:

    >>> nodes1 = [np.array([0.35, 0.9]), np.array([0.0, 0.0]), np.array([0.0])]
    >>> weights1 = [None, np.array([[0.1, 0.4], [0.8, 0.6]]), np.array([0.3, 0.9])]
    >>> network1 = Network(arc1, nodes1, np.array([0.5]), weights1)
    >>> forward_pass(network1)
    >>> calculate_delta(network1, 2)
    >>> network1.error[2]
    array([-0.04068113])
    >>> update_weights(network1, 2)
    >>> calculate_delta(network1, 1)
    >>> network1.error[1]
    array([-0.00240962, -0.00792648])
    """
    if layer == (network.architecture.num_layers-1):
        err = (network.nodes_target - network.nodes[layer]) * (1 - network.nodes[layer]) * network.nodes[layer]
        network.error[layer] = err
    else:
        if network.architecture.layers[-1].size == 1:
            network.error[layer] = np.dot(network.weights[layer+1], network.error[layer+1][0]) * (
                        (1 - network.nodes[layer]) * network.nodes[layer])
            return
        network.error[layer] = np.dot(network.weights[layer+1], network.error[layer+1])*((1-network.nodes[layer])*network.nodes[layer])


def update_weights(network: Network, layer: int):
    """
    Updates the weights of the network.
    :param network: The network
    :param layer: The layer of update the weights at.
    :return:

    >>> layers1 = [Layer(2), Layer(2), Layer(1)]
    >>> nodes1 = [np.array([0.35, 0.9]), np.array([0.0, 0.0]), np.array([0.0])]
    >>> weights1 = [None, np.array([[0.1, 0.4], [0.8, 0.6]]), np.array([0.3, 0.9])]
    >>> network1 = Network(arc1, nodes1, np.array([0.5]), weights1)
    >>> forward_pass(network1)
    >>> calculate_delta(network1, 2)
    >>> update_weights(network1, 2)
    >>> network1.weights[2]
    array([0.27232597, 0.87299836])
    >>> calculate_delta(network1, 1)
    >>> update_weights(network1, 1)
    >>> network1.weights[1]
    array([[0.09915663, 0.39286617],
           [0.79915663, 0.59286617]])
    """
    network.weights[layer] = network.weights[layer]+(network.learning_rate*network.error[layer]*network.nodes[layer-1])


def run(network: Network, iterations=10):
    """
    Runs an iteration
    :param network: The network
    :param iterations: Number of iterations
    :return: The final error (output-target)
    """
    error_network = np.zeros(network.architecture.layers[-1].size)
    for iter in range(iterations):
        forward_pass(network)
        new_error = network.nodes_target - network.nodes[-1]
        if iter != 0 and abs(new_error) > abs(error_network):
            raise ValueError("Local divergence detected")
        error_network = new_error
        for l in range(network.architecture.num_layers - 1, 0, -1):
            calculate_delta(network, l)
            update_weights(network, l)
    return error_network


if __name__ == "__main__":
    # Shareable testdata
    doctest.testmod(
        extraglobs={
            'arc1': Architecture([Layer(2), Layer(2), Layer(1)])
        })


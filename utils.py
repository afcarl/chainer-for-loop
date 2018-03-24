import numpy
import chainer


def make_graph(input_shape, network):
    x = numpy.empty(input_shape, dtype=numpy.float32)
    x = chainer.Variable(x, name='input')
    y = network(x)
    y.name = 'output'
    graph = chainer.computational_graph.build_computational_graph([y])
    return graph


def save_graph(filepath, input_shape, network):
    graph = make_graph(input_shape, network)
    with open(filepath, 'w') as o:
        o.write(graph.dump())

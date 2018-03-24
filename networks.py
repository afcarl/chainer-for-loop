import chainer
import chainer.links as L
import chainer.functions as F


class Exp1(chainer.ChainList):

    def __init__(self, n_layers):
        super(Exp1, self).__init__()
        for layer in range(n_layers):
            self.add_link(L.Linear(None, 2 ** layer))

    def __call__(self, x):
        for link in self.children():
            x = link(x)
        return x


class Exp2(chainer.ChainList):

    def __init__(self, n_layers):
        super(Exp2, self).__init__()
        for layer in range(n_layers):
            self.add_link(L.Linear(None, 2 ** layer))

    def __call__(self, x):
        for link in self.children():
            pre_activate = link(x)
            x = F.relu(pre_activate)
        return pre_activate


class Exp3(chainer.ChainList):

    def __init__(self, n_layers):
        super(Exp3, self).__init__()
        for layer in range(n_layers):
            fc = L.Linear(None, 2 ** layer)
            self.add_link(fc)
            fc.name = 'fc{}'.format(layer)

            if layer != n_layers - 1:
                norm = L.BatchNormalization(2 ** layer)
                self.add_link(norm)
                norm.name = 'norm{}'.format(layer)

    def __call__(self, x):
        for link in self.children():
            pre_activate = link(x)
            if 'fc' in link.name:
                x = pre_activate
            elif 'norm' in link.name:
                x = F.relu(pre_activate)
        return pre_activate


class Exp4(chainer.ChainList):

    def __init__(self, n_layers):
        super(Exp4, self).__init__()
        fc = L.Linear(None, 7 * 7 * 4)
        self.add_link(fc)
        fc.name = 'fc'
        for layer in range(n_layers):
            if layer == n_layers - 1:
                out_channels = 1
            else:
                out_channels = 2 ** layer
            conv = L.Deconvolution2D(
                None, out_channels, ksize=4, stride=2, pad=1)
            self.add_link(conv)
            conv.name = 'conv{}'.format(layer)

            if layer != n_layers - 1:
                norm = L.BatchNormalization(out_channels)
                self.add_link(norm)
                norm.name = 'norm{}'.format(layer)

    def __call__(self, x):
        for link in self.children():
            x = link(x)
            if 'fc' in link.name:
                x = x.reshape(-1, 4, 7, 7)
            if 'norm' in link.name:
                x = F.relu(x)
        return x


class Exp5Enc(chainer.ChainList):

    def __init__(self, n_layers):
        super(Exp5Enc, self).__init__()
        for layer in range(n_layers):
            out_channels = 16 * 2 ** layer
            conv = L.Convolution2D(
                None, out_channels, ksize=4, stride=2, pad=1)
            self.add_link(conv)
            conv.name = 'conv{}'.format(layer)

            if layer != n_layers - 1:
                norm = L.BatchNormalization(out_channels)
                self.add_link(norm)
                norm.name = 'norm{}'.format(layer)

    def __call__(self, x):
        for link in self.children():
            x = link(x)
            if 'norm' in link.name:
                x = F.relu(x)
        return x


class Exp5Dec(chainer.ChainList):

    def __init__(self, n_layers):
        super(Exp5Dec, self).__init__()
        for layer in range(n_layers):
            if layer == n_layers - 1:
                out_channels = 3
            else:
                out_channels = 16 * 2 ** (n_layers - layer - 1)
            conv = L.Deconvolution2D(
                None, out_channels, ksize=4, stride=2, pad=1)
            self.add_link(conv)
            conv.name = 'conv{}'.format(layer)

            if layer != n_layers - 1:
                norm = L.BatchNormalization(out_channels)
                self.add_link(norm)
                norm.name = 'norm{}'.format(layer)

    def __call__(self, x):
        for link in self.children():
            x = link(x)
            if 'norm' in link.name:
                x = F.relu(x)
        return x


class Exp5(chainer.Chain):

    def __init__(self, n_layers):
        super(Exp5, self).__init__()
        with self.init_scope():
            self.enc = Exp5Enc(n_layers)
            self.dec = Exp5Dec(n_layers)

    def __call__(self, x):
        x = self.enc(x)
        x = self.dec(x)
        return x


class Exp6Enc(chainer.ChainList):

    def __init__(self, n_layers):
        super(Exp6Enc, self).__init__()
        for layer in range(n_layers):
            out_channels = 16 * 2 ** layer
            conv = L.Convolution2D(
                None, out_channels, ksize=4, stride=2, pad=1)
            self.add_link(conv)
            conv.name = 'conv{}'.format(layer)

            if layer != n_layers - 1:
                norm = L.BatchNormalization(out_channels)
                self.add_link(norm)
                norm.name = 'norm{}'.format(layer)

    def __call__(self, x):
        features = []
        for link in self.children():
            x = link(x)
            if 'norm' in link.name:
                x = F.relu(x)
                features.append(x)
        return x, features


class Exp6Dec(chainer.ChainList):

    def __init__(self, n_layers):
        super(Exp6Dec, self).__init__()
        for layer in range(n_layers):
            if layer == n_layers - 1:
                out_channels = 3
            else:
                out_channels = 16 * 2 ** (n_layers - layer - 1)
            conv = L.Deconvolution2D(
                None, out_channels, ksize=4, stride=2, pad=1)
            self.add_link(conv)
            conv.name = 'conv{}'.format(layer)

            if layer != n_layers - 1:
                norm = L.BatchNormalization(out_channels)
                self.add_link(norm)
                norm.name = 'norm{}'.format(layer)

    def __call__(self, x, features):
        for link in self.children():
            x = link(x)
            if 'norm' in link.name:
                x = F.relu(x)
                x = F.concat((x, features.pop(-1)))
        return x


class Exp6(chainer.Chain):

    def __init__(self, n_layers):
        super(Exp6, self).__init__()
        with self.init_scope():
            self.enc = Exp6Enc(n_layers)
            self.dec = Exp6Dec(n_layers)

    def __call__(self, x):
        x, features = self.enc(x)
        x = self.dec(x, features)
        return x

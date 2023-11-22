from .mnist_LeNet import MNIST_LeNet, MNIST_LeNet_Autoencoder
from .cifar10_LeNet import CIFAR10_LeNet, CIFAR10_LeNet_Autoencoder
from .cifar10_LeNet_elu import CIFAR10_LeNet_ELU, CIFAR10_LeNet_ELU_Autoencoder
from .mine_net3 import MINE_net3
from .mine_net1D import MINE_net1D
from .mine_netRC2D import MINE_netRC2D
from .mine_netBuffer4D import MINE_netBuffer4D
from .mine_nettwoBuffer2D import MINE_nettwoBuffer2D

def build_network(net_name):
    """Builds the neural network."""


    net = None

    if net_name == 'mnist_LeNet':
        net = MNIST_LeNet()

    if net_name == 'cifar10_LeNet':
        net = CIFAR10_LeNet()

    if net_name == 'cifar10_LeNet_ELU':
        net = CIFAR10_LeNet_ELU()

    if net_name == 'mine_net3':
        net = MINE_net3()    
        
    if net_name == 'mine_net1D':
        net = MINE_net1D()

    if net_name == 'mine_netRC2D':
        net = MINE_netRC2D()

    if net_name == 'mine_netBuffer4D':
        net = MINE_netBuffer4D()

    if net_name == 'mine_nettwoBuffer6P':
        net = MINE_nettwoBuffer2D()
                
    return net


def build_autoencoder(net_name):
    """Builds the corresponding autoencoder network."""

    implemented_networks = ('mnist_LeNet', 'cifar10_LeNet', 'cifar10_LeNet_ELU')
    assert net_name in implemented_networks

    ae_net = None

    if net_name == 'mnist_LeNet':
        ae_net = MNIST_LeNet_Autoencoder()

    if net_name == 'cifar10_LeNet':
        ae_net = CIFAR10_LeNet_Autoencoder()

    if net_name == 'cifar10_LeNet_ELU':
        ae_net = CIFAR10_LeNet_ELU_Autoencoder()

    return ae_net

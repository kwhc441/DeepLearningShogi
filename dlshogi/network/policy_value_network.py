import torch
import torch.nn as nn
import re

def policy_value_network(network, add_sigmoid=False):
    # wideresnet10 and resnet10_swish are treated specially because there are published models
    if network == 'wideresnet10':
        from dlshogi.network.policy_value_network_wideresnet10 import PolicyValueNetwork
    elif network == 'resnet10_swish':
        from dlshogi.network.policy_value_network_resnet10_swish import PolicyValueNetwork
    elif network[:6] == 'resnet':
        from dlshogi.network.policy_value_network_resnet import PolicyValueNetwork
    elif network[:5] == 'senet':
        from dlshogi.network.policy_value_network_senet import PolicyValueNetwork
    elif network[:8] == 'convnext':
        from dlshogi.network.policy_value_network_convnext import PolicyValueNetwork
    else:
        # user defined network
        names = network.split('.')
        if len(names) == 1:
            PolicyValueNetwork = globals()[names[0]]
        else:
            from importlib import import_module
            PolicyValueNetwork = getattr(import_module('.'.join(names[:-1])), names[-1])

    if add_sigmoid:
        class PolicyValueNetworkAddSigmoid(PolicyValueNetwork):
            def __init__(self, *args, **kwargs):
                super(PolicyValueNetworkAddSigmoid, self).__init__(*args, **kwargs)

            def forward(self, x1, x2):
                y1, y2 = super(PolicyValueNetworkAddSigmoid, self).forward(x1, x2)
                return y1, torch.sigmoid(y2)

        PolicyValueNetwork = PolicyValueNetworkAddSigmoid

    if network in [ 'wideresnet10', 'resnet10_swish' ]:
        return PolicyValueNetwork()
    elif network[:6] == 'resnet' or network[:5] == 'senet' or network[:8] == 'convnext':
        m = re.match('^(resnet|senet|convnext)(\d+)(x\d+){0,1}(_fcl\d+){0,1}(_reduction\d+){0,1}(_relu|_swish|_gelu){0,1}(_kernel\d+){0,1}$', network)

        # blocks
        blocks = int(m[2])

        # channels
        if m[3] is None:
            channels = { 10: 192, 15: 224, 20: 256 }[blocks]
        else:
            channels = int(m[3][1:])

        # fcl
        if m[4] is None:
            fcl = 256
        else:
            fcl = int(m[4][4:])

        # activation
        if m[6] is None:
            if m[1] != 'convnext':
                activation = nn.GELU()
            else:
                activation = nn.ReLU()
        else:
            activation = { '_relu': nn.ReLU(), '_swish': nn.SiLU(), '_gelu': nn.GELU() }[m[6]]

        if m[1] == 'resnet':
            return PolicyValueNetwork(blocks=blocks, channels=channels, activation=activation, fcl=fcl)
        elif m[1] == 'senet':
            # reduction
            if m[5] is None:
                reduction = 8
            else:
                reduction = int(m[5][10:])
            return PolicyValueNetwork(blocks=blocks, channels=channels, activation=activation, fcl=fcl, reduction=reduction)
        elif m[1] == 'convnext':
            if m[7] is None:
                kernel = 3
            else:
                kernel = int(m[7][7:])
            return PolicyValueNetwork(blocks=blocks, channels=channels, activation=activation, fcl=fcl, kernel=kernel)
    else:
        return PolicyValueNetwork()

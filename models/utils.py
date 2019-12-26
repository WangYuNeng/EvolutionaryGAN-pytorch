from collections import namedtuple, OrderedDict

import torch
import numpy as np

from util.minheap import MinHeap


G_Net = namedtuple(
    "G_Net",
    "fitness G_candis optG_candis losses",
)


def get_G_heap(candi_num):
    return MinHeap([
        G_Net(
            fitness=-float('inf'),
            G_candis=None,
            optG_candis=None,
            losses=None,
        )
        for _ in range(candi_num)
    ])


def categorize_mappings(networks, optimizers):
    mappings = [network['module.layer'] for network in networks]
    SO_mappings, non_SO_mappings, SO_optG, non_SO_optG = [], [], [], []
    for mapping, optimizer in zip(mappings, optimizers):
        if mapping.det().item() > 0:
            SO_mappings.append(mapping)
            SO_optG.append(optimizers)
        else:
            non_SO_mappings.append(mapping)
            non_SO_optG.append(optimizer)
    return SO_mappings, non_SO_mappings, SO_optG, non_SO_optG


def combine_mapping_networks(*mappings, is_SO: bool):
    """
    get skew-symmetric matrix by Cayley transform
    https://en.wikipedia.org/wiki/Cayley_transform
    then interpolate between
    """
    assert len(mappings) == 2
    I = torch.eye(*mappings[0].shape, device=mappings[0].device)
    if not is_SO:
        for i in range(2):
            mappings[i][0] *= -1
    skews = [(I - mapping) @ (I + mapping).pinverse() for mapping in mappings]
    child_skew = (skews[0] + skews[1]) / 2  # interpolate
    child_mapping = (I - child_skew) @ (I + child_skew).pinverse()
    if not is_SO:
        child_mapping[0] *= -1
        for i in range(2):
            mappings[i][0] *= -1
    return OrderedDict({'module.layer': child_mapping})


def coshuffle(*lists):
    length = len(lists[0])
    new_lists = [[] for _ in lists]
    shuffle_indices = np.random.permutation(length)
    for index in shuffle_indices:
        for list_idx, l in enumerate(lists):
            new_lists[list_idx].append(l[index])
    return new_lists

import numpy
import rpack


def max_depth(t):
    if len(t.children) == 0:
        return 0
    return numpy.max([max_depth(c) for c in t.children]) + 1


def at_depth(t, depth, property=None):
    if depth == 0:
        return [t.data.get(property, t)]
    ret = []
    for c in t.children:
        ret.extend(at_depth(c, depth - 1, property=property))
    return ret


def at_max_depth(t, property=None):
    return at_depth(t, max_depth(t), property=property)


def leaves(t, property=None):
    if len(t.children) == 0:
        return [t.data.get(property, t)]
    ret = []
    for c in t.children:
        ret.extend(leaves(c, property=property))
    return ret


def deep_copy(t):
    from voxcell import Hierarchy
    h_out = Hierarchy(t.data.copy())
    for c in t.children:
        h_out.children.append(deep_copy(c))
    return h_out

def hierarchy_to_dict(h):
    ret = {}
    ret.update(h.data)
    ret["children"] = [hierarchy_to_dict(c) for c in h.children]
    return ret    


def normalization_spread(annotation, hierarchy, region):
    constant = 12
    voxel_index = list(hierarchy.collect('acronym', region, 'id'))
    count = 0
    for i in range(len(voxel_index)):
        count += numpy.count_nonzero(annotation == voxel_index[i])
    norm_arg = round(numpy.sqrt(count/constant))
    return int(norm_arg)      

def normalization_offsets(lst_spread, configuration_json):
    squares = []
    for spread in lst_spread:
        squares.append((spread, spread))
    positions = rpack.pack(squares)
    for i in range(len(configuration_json)):
        configuration_json[i]['normalization_args']['normalize_offsets'] = list(
            positions[i])
    return configuration_json
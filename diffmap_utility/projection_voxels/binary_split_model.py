import numpy
import os

from scipy.spatial import distance

from projection_voxels.cached_projection import CachedProjections

def rotmat(angle):
    return numpy.array([
        [numpy.cos(angle), -numpy.sin(angle)],
        [numpy.sin(angle), numpy.cos(angle)]
    ])

def angle_vector(angle):
    return numpy.array([numpy.cos(angle), numpy.sin(angle)])

def random_angle(probs):
    c = numpy.cumsum(probs[:, 0])
    c = c / c[-1]
    return numpy.interp(numpy.random.rand(), c, probs[:, 1])


class Region(object):
    PARAMETER_DICT = {
        "_pt": "anchor_point",
        "_nrml": "normal_vector",
        "_id": "id",
        "_sz": "size"
    }
    def __init__(self, pt, nrml, parent=None):
        self._pt = pt.reshape((1, -1))
        self._nrml = nrml.reshape((1, -1))
        self._children = []
        self._parent = parent
        self._id = None
        self._sz = 0
    
    @property
    def as_dict(self):
        d = dict([
            (v, self.__dict__[k])
            for k, v in Region.PARAMETER_DICT.items()
        ])
        for k, v in d.items():
            if isinstance(v, numpy.ndarray):
                d[k] = [list(map(float, _v)) for _v in v]
        d["children"] = [c.as_dict for c in self._children]
        return d
    
    @property
    def hierarchy_dict(self):
        return {
            "name": "Binary split region {0}".format(self._id),
            "acronym": "BSP{0}".format(self._id),
            "id": self._id,
            "children": [c.hierarchy_dict for c in self._children]
        }
    
    @property
    def is_leaf(self):
        return len(self._children) == 0
    
    @property
    def is_root(self):
        return self._parent is None
    
    @property
    def nrml(self):
        return self._nrml
    
    @property
    def tangent(self):
        if self.nrml is None: return None
        if (numpy.arctan2(*self.nrml[0]) - numpy.arctan2(*self._parent.nrml[0])) < 0:
            return numpy.dot(self.nrml, rotmat(-numpy.pi/2))
        return numpy.dot(self.nrml, rotmat(numpy.pi/2))
    
    @staticmethod
    def distance_dependent_connectivity(D, offset, con_range):
        return numpy.exp(-(D + offset) / con_range) / numpy.exp(-offset / con_range)
    
    def test_contained(self, pts, propagate=True):
        ret = numpy.dot(self._nrml, (pts - self._pt).transpose())[0] >= 0
        if propagate:
            if not self.is_root:
                ret = ret & self._parent.test_contained(pts, propagate=True)
        return ret
    
    def predetermined_split(self, splt_dict):
        for c_params in splt_dict["children"]:
            c = Region(numpy.array(c_params[Region.PARAMETER_DICT["_pt"]]),
                       numpy.array(c_params[Region.PARAMETER_DICT["_nrml"]]),
                       parent=self)
            c._sz = c_params[Region.PARAMETER_DICT["_sz"]]
            c._id = c_params[Region.PARAMETER_DICT["_id"]]
            c.predetermined_split(c_params)
            self._children.append(c)
    
    def split(self, seeds, offset=0, propagate=2000, angle_range=0.3):
        self_a = numpy.arctan2(self._nrml[0, 1], self._nrml[0, 0])
        p_vec = numpy.array([[0.0, self_a + numpy.pi/2 - angle_range],
                             [1.0, self_a + numpy.pi/2 + angle_range]])
        a = random_angle(p_vec)
        tg = self._parent._children[0].tangent
        s = self._pt - self.y(seeds).mean() * tg + self.x(seeds).mean() * self._nrml
        return self._split(s[0], a, seeds, offset, propagate)
    
    def _split(self, s, a, seeds, offset, propagate, angle_range=0.3):
        self._id = offset
        offset += 1
        self._children.append(Region(s, angle_vector(a), parent=self))
        self._children.append(Region(s, angle_vector(a + numpy.pi), parent=self))
        
        v = self._children[0].test_contained(seeds, propagate=False)
        self._children[0]._sz = float(v.sum())
        self._children[1]._sz = float((~v).sum())
        if v.sum() >= propagate:
            offset = self._children[0].split(seeds[v], propagate=propagate,
                                             offset=offset, angle_range=angle_range)
        else:
            self._children[0]._id = offset; offset += 1
        if (~v).sum() >= propagate:
            offset = self._children[1].split(seeds[~v], propagate=propagate,
                                             offset=offset, angle_range=angle_range)
        else:
            self._children[1]._id = offset; offset += 1
        return offset
    
    def plot(self):
        from matplotlib import pyplot as plt
        plt.plot(self._pt[0, 0], self._pt[0, 1], 'ro')
        plt.plot([self._pt[0, 0], self._pt[0, 0] + 10 * self._nrml[0, 0]],
                 [self._pt[0, 1], self._pt[0, 1] + 10 * self._nrml[0, 1]],
                color="red")
        
        tangent = self.tangent
        ln = self._pt + numpy.linspace(-100, 100, 500).reshape((-1, 1)) * tangent
        if not self.is_root:
            v = self._parent.test_contained(ln)
        plt.plot(ln[v, 0], ln[v, 1], color="black")
        
        for c in self._children:
            c.plot()
    
    def resolve(self, pts):
        ret = -numpy.ones(len(pts), dtype=int)
        if self.is_leaf:
            ret[self.test_contained(pts)] = self._id
            return ret
        v = self._children[0].test_contained(pts)
        ret[v] = self._children[0].resolve(pts[v])
        ret[~v] = self._children[1].resolve(pts[~v])
        return ret
    
    def lowest_common_ancestor(self, i, j):
        if self.is_leaf:
            if self._id == i: return [0]
            if self._id == j: return [1]
            return []
        a = self._children[0].lowest_common_ancestor(i, j)
        b = self._children[1].lowest_common_ancestor(i, j)
        if isinstance(a, Region): return a
        if isinstance(b, Region): return b
        if len(a + b) == 2: return self
        return a + b
    
    def x(self, pts):
        pts_o = pts - self._pt
        return numpy.dot(self.nrml, pts_o.transpose())[0]
    
    def y(self, pts):
        ret = self._parent.x(pts)
        ret = ret - self._parent.x(self._pt)
        return ret
    
    def converted(self, pts):
        return numpy.vstack([self.x(pts), self.y(pts)]).transpose()
    
    def inverted(self, pts):
        return pts[:, 0:1] * self.nrml + pts[:, 1:2] * self._parent.nrml + self._pt
    
    def _transpose(self, pts, at_depth=0):
        if self.is_leaf: return {at_depth: pts}
        ret = numpy.NaN * numpy.ones_like(pts)
        v_pts = self._children[0].test_contained(pts, propagate=False)
        ret[v_pts, :] = self._children[1].inverted(self._children[0].converted(pts[v_pts]))
        ret[~v_pts, :] = self._children[0].inverted(self._children[1].converted(pts[~v_pts]))
        
        ll = self._children[0]._transpose(pts[v_pts], at_depth=at_depth + 1)
        lr = self._children[1]._transpose(pts[~v_pts], at_depth=at_depth + 1)
        
        res = {at_depth: ret}
        for k, v in ll.items():
            res.setdefault(k, numpy.NaN * numpy.ones_like(pts))[v_pts, :] = v
        for k, v in lr.items():
            res.setdefault(k, numpy.NaN * numpy.ones_like(pts))[~v_pts, :] = v
        return res
    
    def transpose(self, pts):
        return self._transpose(pts, 0)

class RootRegion(Region):
    STR_GLOBALS = "global_limits"
    def __init__(self, global_lims):
        self._globals = global_lims
        X, Y = numpy.meshgrid(range(*global_lims[0]), range(*global_lims[1]))
        self._seeds = numpy.hstack([X.reshape((-1, 1)), Y.reshape((-1, 1))])
        self._children = []
        self._id = None
        self._sz = 0
        self._pt = None
        self._nrml = None
    
    @classmethod
    def from_dict(cls, d):
        obj = cls(numpy.array(d[RootRegion.STR_GLOBALS]))
        obj._pt = numpy.array(d[Region.PARAMETER_DICT["_pt"]])
        obj.predetermined_split(d)
        return obj
    
    @property
    def as_dict(self):
        d = super().as_dict
        d[RootRegion.STR_GLOBALS] = [list(map(int, _g)) for _g in self._globals]
        return d

    @property
    def hierarchy_dict(self):
        return {
            "name": "Binary split root",
            "acronym": "BSPRoot",
            "id": self._id,
            "children": [c.hierarchy_dict for c in self._children]
        }
    
    def test_contained(self, ln, propagate=False):
        return ((ln[:, 0] >= self._globals[0][0]) & (ln[:, 0] < self._globals[0][1]) &
                (ln[:, 1] >= self._globals[1][0]) & (ln[:, 1] < self._globals[1][1]))
    
    def split(self, pt, offset=0, propagate=1000, angle_range=0.3):
        self._pt = pt.reshape((1, -1))
        self._id = offset
        self._sz = float(len(self._seeds))
        offset += 1
        a = random_angle(numpy.array([[0, 0], [1, numpy.pi]]))
        return self._split(pt, a, self._seeds, offset, propagate, angle_range=angle_range)
    
    def plot(self):
        from matplotlib import pyplot as plt
        for c in self._children:
            c.plot()
        plt.gca().set_xlim(self._globals[0])
        plt.gca().set_ylim(self._globals[1])
    
    def lowest_common_ancestor(self, i, j):
        if self.is_leaf:
            return []
        return super().lowest_common_ancestor(i, j)
    
    @property
    def nrml(self):
        lower_a = numpy.arctan2(self._children[0]._nrml[0, 1],
                                self._children[0]._nrml[0, 0])
        a = lower_a + numpy.pi / 2
        vec = angle_vector(a).reshape((1, -1))
        return vec
    
    def connect(self, pts_fr, pts_to, con_params):
        from scipy.spatial.distance import cdist
        if self.is_leaf:
            return self.distance_dependent_connectivity(cdist(pts_fr, pts_to), con_params)
        
        centers = self.transpose(pts_fr)
        e_sz = (len(pts_fr), len(pts_to))
        P = numpy.zeros(e_sz)

        for k in centers.keys():
            v = centers[k]
            strength = numpy.maximum(float(con_params["str_max"]) - k, 1) ** con_params["str_exponent"]
            p_range = con_params["range_max"] * numpy.exp(-k / con_params["range_div"])
            D = distance.cdist(v, pts_to)
            added_P = Region.distance_dependent_connectivity(D, con_range=p_range, offset=con_params["offset"]) * strength
            valid = ~numpy.isnan(added_P)
            P[valid] = P[valid] + added_P[valid]

        return P

class BinarySplitVoxelArray(object):
    def __init__(self, root, depth, connectivity_params):
        self._root = root
        self._depth = depth
        self._con_params = connectivity_params
    
    def initialize_split(self, rnd_seed):
        numpy.random.seed(rnd_seed)
        rnd = 0.25 * (numpy.random.rand(2) - 0.5)
        first_point = self._root._globals.mean(axis=1) + rnd * numpy.diff(self._root._globals, axis=1)[:, 0]
        self._root.split(first_point)
    
    @classmethod
    def from_json(cls, fn):
        import json
        with open(fn, "r") as fid:
            d = json.load(fid)
        root = RootRegion.from_dict(d["regions"])
        return cls(root, d["depth"], d["connectivity_parameters"])
    
    def save_json(self, fn):
        import json
        d = {
            "regions": self._root.as_dict,
            "depth": int(self._depth),
            "connectivity_parameters": self._con_params
        }
        with open(fn, "w") as fid:
            json.dump(d, fid, indent=2)
    
    def save_atlas(self, fn):
        from voxcell import VoxelData
        annotations = VoxelData(self.true_annotations, (1, 1, 1), offset=(0.0, 0.0, 0.0))
        annotations.save_nrrd(fn)
    
    def save_hierarchy(self, fn):
        import json
        with open(fn, "w") as fid:
            json.dump(self.hierarchy_dict, fid)
    
    @property
    def shape(self):
        shape = (self._root._globals[0][1] - self._root._globals[0][0],
                 self._root._globals[1][1] - self._root._globals[1][0],
                 self._depth)
        return shape
    
    @property
    def source_coords_3d(self):
        Y, X, Z = numpy.meshgrid(range(self._root._globals[1][0], self._root._globals[1][1]),
                                 range(self._root._globals[0][0], self._root._globals[0][1]),
                                 range(self._depth))
        return numpy.vstack([X.flatten(), Y.flatten(), Z.flatten()]).transpose()
    
    @property
    def target_coords_3d(self):
        return self.source_coords_3d
    
    @property
    def true_annotations(self):
        M = self._root.resolve(self.source_coords_3d[:, :2]).reshape(self.shape)
        return M
    
    @property
    def hierarchy_dict(self):
        return self._root.hierarchy_dict
    
    def projection_strengths(self, source, target):
        source = numpy.array(source)
        if source.ndim < 2: source = source.reshape((-1, 2))
        target = numpy.array(target)
        if target.ndim < 2: target = target.reshape((-1, 2))

        P = self._root.connect(source, target, self._con_params)
        return P + numpy.random.rand(*P.shape) * self._con_params["noise"]
    
    def __getitem__(self, idx):
        fr, to = idx
        src_pts = self.source_coords_3d[fr][:, :2]
        tgt_pts = self.target_coords_3d[to][:, :2]
        return self.projection_strengths(src_pts, tgt_pts)


class BinarySplitModel(CachedProjections):
    FN_MODEL = "b_model.json"
    FN_ANNOTATIONS = "b_annotations.nrrd"
    FN_HIERARCHY = "b_hierarchy.json"

    def __init__(self, root, voxel_sizes=(100.0, 100.0, 100.0),
                 fn_annotations=None, fn_hierarchy=None,
                 cache_file=None, grow_cache=True):
        fn_model = os.path.join(root, BinarySplitModel.FN_MODEL)
        if fn_annotations is None: fn_annotations = os.path.join(root, BinarySplitModel.FN_ANNOTATIONS)
        if fn_hierarchy is None: fn_hierarchy = os.path.join(root, BinarySplitModel.FN_HIERARCHY)
        
        voxel_array = BinarySplitVoxelArray.from_json(fn_model)
        source_coords_3d = voxel_array.source_coords_3d
        target_coords_3d = voxel_array.target_coords_3d
        region_annotation_args = (fn_annotations, )
        hierarchy_args = (fn_hierarchy, )
        
        super().__init__(voxel_array, source_coords_3d, target_coords_3d,
                         region_annotation_args, hierarchy_args, cache_file, grow_cache)
        self._voxel_sizes = voxel_sizes
    
    def _three_d_indices_to_output_coords(self, idx, direction):  # As in aibs_mcm_projection
        return numpy.array(idx) * numpy.array([self._voxel_sizes])

import os

from .parcellation_level import ParcellationLevel


class BinarySplitLevel(ParcellationLevel):
    def __init__(self, root, config, hierarchy, structures, region_volume, overwrite=False):
        super().__init__(root, config, hierarchy, region_volume, overwrite)
        self._structures = structures

    @property
    def cache_cfg(self):
        cache_cfg = {
            "class": "BinarySplitModel",
            "args": {
                "ToyModelRoot": os.path.abspath(os.path.split(self._config["inputs"]["anatomical_model"])[0]),
                "AnnotationFile": self.region_volume_fn,
                "HierarchyFile": self.hierarchy_fn,
                "H5Cache": os.path.join(os.path.split(self._config["inputs"]["anatomical_flatmap"])[0],
                                        "b_projections_h5_cache.h5")
            }
        }
        return cache_cfg
    
    @property 
    def structures(self):
        if self._structures is not None: return self._structures
        structures = []
        df = self.hierarchy.as_dataframe()

        def r(i):
            if df["parent_id"][i] < 0:
                return 0
            return r(df["parent_id"][i]) + 1
        def s(i):
            if df["parent_id"][i] < 0:
                return [int(i)]
            return s(df["parent_id"][i]) + [int(i)]

        for r_id, row in df.iterrows():
            structures.append({
                "acronym": row["acronym"],
                "name": row["name"],
                "id": int(r_id),
                "graph_id": int(1),
                "graph_order": int(r(r_id)),
                "structure_id_path": s(r_id),
                "structure_set_ids": [int(0)]
            })
        return structures
    
    @staticmethod
    def find_inputs_from_file_system(root, config):
        h, v = ParcellationLevel.find_inputs_from_file_system(root, config)
        return h, None, v
    
    @staticmethod
    def find_inputs_from_config(config, initial_parcellation):
        custom_hierarchy, annotations = ParcellationLevel.find_inputs_from_config(config, initial_parcellation)
        return custom_hierarchy, None, annotations

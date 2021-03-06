import json
import collections
import pickle

import gridfs
from bson import ObjectId
from pymongo import MongoClient


class SacredExperimentAccess(object):

    def __init__(self, database_name='sacred', host='localhost', port=9001):
        client = MongoClient(host, port)
        self.database = client[database_name]
        self.fs = gridfs.GridFS(self.database)

    @staticmethod
    def filter_from_nested_dict(d, base_key=""):
        ret = {}
        for k, v in d.items():
            if isinstance(v, dict):
                sub_vals = SacredExperimentAccess.filter_from_nested_dict(v, f"{base_key}{k}.")
                ret = {**ret, **sub_vals}
            else:
                ret[base_key + k] = v
        return ret

    def get_experiments(self, name=None, complete=False, config={}, additional_filter=None):
        filter = {}
        if name:
            filter['experiment.name'] = name
        if complete:
            filter['status'] = {'$in': ['COMPLETED']}
        if config:
            # config will be a dictionary
            filter = {**filter, **SacredExperimentAccess.filter_from_nested_dict(config, "config.")}
        if additional_filter:
            filter = {**filter, **additional_filter}

        return list(self.database.runs.find(filter))

    def load_artifacts(self, objects):
        '''
        Return the full experiment objects associated with the objects.
        :param objects: Can be one of: A single experiments dict. An iterable of experiments
        :returns The same objects, with the artifacts loaded in. Assumes all artifacts are JSON files
        '''
        if isinstance(objects, dict):
            for i, artifact in enumerate(objects['artifacts']):
                if ".json" in artifact["file_id"]:
                    objects['artifacts'][i]['object'] = json.loads(self.fs.get(artifact['file_id']).read())
                else:
                    objects['artifacts'][i]['object'] = json.loads(self.fs.get(artifact['file_id']).read())
        elif isinstance(objects, collections.Iterable):
            for object in objects:
                for i, artifact in enumerate(object['artifacts']):
                    if ".json" in artifact["file_id"]:
                        objects['artifacts'][i]['object'] = pickle.load(self.fs.get(artifact['file_id']).read())
                    else:
                        objects['artifacts'][i]['object'] = pickle.loads(self.fs.get(artifact['file_id']).read())
        return objects

    def get_artifacts_by_id(self, ids):
        '''
        Return the artifacts associated with the ids.
        :param objects: A list of ObjectIds
        '''

        return [self.fs.get(id).read() for id in ids]

    def get_metrics_by_exp(self, exp_objects, metric_names, filter_type="equals"):
        """
        First layer: which experiment, second index: which metric.

        :param exp_objects: dict of iterable of dicts
        :param metric_names: metrics to get
        :return:
        """
        def get_metrics_for_exp(exp, names, filter_type=filter_type):
            metrics = []
            for n in names:
                if filter_type == "equals":
                    metrics.extend(list(filter(lambda x: x["name"] == n, exp["info"]["metrics"])))
                elif filter_type == "in":
                    metrics.extend(list(filter(lambda x: n in x["name"], exp["info"]["metrics"])))
            metric_objects = [list(self.database.metrics.find({"_id": ObjectId(x["id"])}))[0] for x in metrics]
            return metric_objects

        if isinstance(metric_names, str):
            metric_names = [metric_names]

        if isinstance(exp_objects, dict):
            return get_metrics_for_exp(exp_objects, metric_names)

        elif isinstance(exp_objects, collections.Iterable):
            ret = []
            for exp in exp_objects:
                ret.append(get_metrics_for_exp(exp, metric_names))
            return ret


def get_dicts_key_subset(dicts, keys):
    ret = []
    for d in dicts:
        ret.append({k: d[k] for k in keys})
    return ret


def get_unique_dicts(dicts):
    ret = []
    for dict in dicts:
        ret.append(json.dumps(dict, sort_keys=True))
    ret = list(set(ret))
    return [json.loads(r) for r in ret]


def flatten_dict(d, parent_key='', sep='.'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

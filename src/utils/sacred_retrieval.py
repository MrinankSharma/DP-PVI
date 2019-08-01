import json
from collections import Iterable

import gridfs
from pymongo import MongoClient


class SacredExperimentAccess(object):

    def __init__(self, database_name='sacred', host='localhost', port=9001):
        client = MongoClient(host, port)
        self.database = client[database_name]
        self.fs = gridfs.GridFS(self.database)

    def get_experiments(self, name=None, complete=False, config=None, additional_filter=None):
        filter = {}
        if name:
            filter['experiment.name'] = name
        if complete:
            filter['status'] = 'COMPLETED'
        if config:
            filter['config'] = config
        if additional_filter:
            filter = {**filter, **additional_filter}

        return list(self.database.runs.find(filter))

    def load_artifacts(self, objects):
        '''
        Return the artifacts associated with the objects.
        :param objects: Can be one of: A single experiments dict. An iterable of experiments
        :returns The same objects, with the artifacts loaded in. Assumes all artifacts are JSON files
        '''
        if isinstance(objects, dict):
            for i, artifact in enumerate(objects['artifacts']):
                objects['artifacts'][i]['object'] = json.loads(self.fs.get(artifact['file_id']).read())
        elif isinstance(objects, Iterable):
            for object in objects:
                for i, artifact in enumerate(object['artifacts']):
                    object['artifacts'][i]['object'] = json.loads(self.fs.get(artifact['file_id']).read())

        return objects

    def get_artifacts_by_id(self, ids):
        '''
        Return the artifacts associated with the ids.
        :param objects: A list of ObjectIds
        '''

        return [self.fs.get(id).read() for id in ids]


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

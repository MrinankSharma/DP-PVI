from ruamel.yaml import YAML
from ruamel.yaml.compat import StringIO
import numpy


class YAMLStringDumper(YAML):

    def __init__(self):
        super().__init__()
        self.indent(mapping=2, sequence=2, offset=2)

    def dump(self, data, stream=None, **kw):
        prepped_data = YAMLStringDumper.prep_data(data)
        inefficient = False
        if stream is None:
            inefficient = True
            stream = StringIO()
        YAML.dump(self, prepped_data, stream, **kw)
        if inefficient:
            return stream.getvalue()

    @staticmethod
    def prep_data(data):
        if isinstance(data, dict):
            ret = {}
            for key, val in data.items():
                if isinstance(val, numpy.ndarray):
                    ret[key] = val.tolist()
                else:
                    ret[key] = val
            return ret
        else:
            return data

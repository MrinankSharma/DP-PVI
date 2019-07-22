import logging

import requests
from ruamel.yaml import YAML
from ruamel.yaml.compat import StringIO

DEFAULT_WEBHOOK_LOCATION = "../../slack_webhook"


class MyYAML(YAML):
    def dump(self, data, stream=None, **kw):
        inefficient = False
        if stream is None:
            inefficient = True
            stream = StringIO()
        YAML.dump(self, data, stream, **kw)
        if inefficient:
            return stream.getvalue()


yaml = MyYAML()  # or typ='safe'/'unsafe' etc

logger = logging.getLogger(__name__)
yaml = MyYAML()
yaml.indent(mapping=8, sequence=2, offset=8)


def slack_notification(experiment_tag, content, slack_webhook_file=DEFAULT_WEBHOOK_LOCATION):
    """
    Send slack notification to a webhook

    Example Usage:

        experiment_tag = "My Experiment Data"
        metadata = {
            "seed": 0,
            "running_from_pycharm": "True",
            "nested_metadata": {
                "name": "metadata",
                "ints": [1, 2, 3]
            }
        }

        results = {
            "loss": 0.1,
            "epoch": 0.25
        }

        slack_notification(experiment_tag, ["*Results*", results, "*Metadata*", metadata])

    """
    try:
        with open(slack_webhook_file, 'r') as file:
            webhook_location = file.read().replace('\n', '')
    except FileNotFoundError:
        logger.error("Slack Webhook File Not Found - Notification Skipped")
        return

    text_str = "*{}*\n\n".format(experiment_tag)

    if type(content) == list:
        for content_obj in content:
            text_str = text_str + content_obj_to_str(content_obj) + "\n\n"
    else:
        text_str = text_str + content_obj_to_str(content)

    payload = {
        "text": text_str
    }

    requests.post(webhook_location, json=payload)


def content_obj_to_str(content_obj):
    if type(content_obj) == dict:
        return yaml.dump(content_obj)
    elif type(content_obj) == str:
        return content_obj

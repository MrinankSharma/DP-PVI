from sacred.commandline_options import CommandLineOption
from sacred.observers import MongoObserver, SlackObserver
import logging

logger = logging.getLogger(__name__)

"""
Custom flags for database logging
"""


class TestMongoDbOption(CommandLineOption):
    """
    Run using a database called test in the standard location
    """
    short_flag = 'test'

    @classmethod
    def apply(cls, args, run):
        # run.config contains the configuration. You can read from there.
        MongoObserver.create(url='my.server.org:27017',
                             db_name='MY_DB')
        mongo = MongoObserver.create(url="localhost:9001", db_name='test')
        run.observers.append(mongo)
        logger.info("Saving to database test WITHOUT slack notifications")
        run.info = {
            **run.info,
            "test": True
        }


class ExperimentMongoDbOption(CommandLineOption):
    """
    Use the sacred database.
    """
    short_flag = 'run'

    @classmethod
    def apply(cls, args, run):
        # run.config contains the configuration. You can read from there.
        MongoObserver.create(url='my.server.org:27017',
                             db_name='MY_DB')
        mongo = MongoObserver.create(url="localhost:9001", db_name='sacred')
        run.observers.append(mongo)
        run.observers.append(
            SlackObserver.from_config('../../slack.json')
        )
        logger.info("Saving to database sacred")

        run.info = {
            **run.info,
            "test": False
        }
from sacred.commandline_options import CommandLineOption
from sacred.observers import MongoObserver, SlackObserver
from src.utils.sacred_retrieval import SacredExperimentAccess
import sys
import logging

logger = logging.getLogger(__name__)

"""
Custom flags for database logging
"""


class TestOption(CommandLineOption):
    """
    Run using a database called test in the standard location
    """
    short_flag = 'tr'

    @classmethod
    def apply(cls, args, run):
        a = SacredExperimentAccess(database_name="test")
        if len(a.get_experiments(config=run.config)) > 0:
            logger.info("Experiment has already been run - don't bother!")
            sys.exit()
        mongo = MongoObserver.create(url="localhost:9001", db_name='test')
        run.observers.append(mongo)
        logger.info("Saving to database test WITHOUT slack notifications")
        run.info = {
            **run.info,
            "test": True
        }


class ExperimentOption(CommandLineOption):
    """
    Use the sacred database.
    """
    short_flag = 'fr'

    @classmethod
    def apply(cls, args, run):
        a = SacredExperimentAccess()
        if len(a.get_experiments(name="jalko2017", config=run.config, complete=True)) > 0:
            logger.info("Experiment has already been run - don't bother!")
            sys.exit()

        # run.config contains the configuration. You can read from there.
        mongo = MongoObserver.create(url="localhost:9001", db_name='sacred')
        run.observers.append(mongo)
        run.observers.append(
            SlackObserver.from_config(run.config["slack_json_file"])
        )
        logger.info("Saving to database sacred")

        run.info = {
            **run.info,
            "test": False
        }

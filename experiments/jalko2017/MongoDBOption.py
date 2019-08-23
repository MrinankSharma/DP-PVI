import logging
import sys

from sacred.commandline_options import CommandLineOption
from sacred.observers import MongoObserver, SlackObserver

from src.utils.sacred_retrieval import SacredExperimentAccess

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
        mongo = MongoObserver.create(url="localhost:9001", db_name='test')
        logging.basicConfig()
        logging.getLogger().setLevel(logging.DEBUG)
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
        if len(a.get_experiments(config=run.config, complete=True)) > 0:
            logger.info("Experiment has already been run - don't bother!")
            logger.info("Note that this will **not** show up in sacred")
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


class DatabaseOption(CommandLineOption):
    """
    Use the sacred database.
    """
    short_flag = 'c'
    arg = 'sacred'

    @classmethod
    def apply(cls, args, run):
        a = SacredExperimentAccess()
        if len(a.get_experiments(config=run.config, complete=True)) > 0:
            logger.info("Experiment has already been run - don't bother!")
            logger.info("Note that this will **not** show up in sacred")
            sys.exit()

        # run.config contains the configuration. You can read from there.
        mongo = MongoObserver.create(url="localhost:9001", db_name=args)
        run.observers.append(mongo)
        run.observers.append(
            SlackObserver.from_config(run.config["slack_json_file"])
        )
        logger.info(f"Saving to database {args}")

        run.info = {
            **run.info,
            "test": False
        }

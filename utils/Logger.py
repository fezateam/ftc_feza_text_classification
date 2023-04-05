import logging
import logging.config
import yaml
import datetime


def logger() -> logging:
    """
    The logger function is a Python function that returns a configured logger object for logging purposes. The function performs the following tasks:
    Generates a timestamped filename for the log file based on the current date and time using the now() function from an unknown module.
    Reads a YAML configuration file for logging settings from "./configs/log.yaml" and loads it using the yaml.safe_load() function from the PyYAML module.
    Modifies the logging configuration dictionary by setting the "filename" field to the generated filename.
    Configures the logging system using the logging.config.dictConfig() function from the Python standard library, passing in the modified configuration dictionary.
    Retrieves the logger object using the logging.getLogger(__name__) method from the Python standard library.
    Returns the configured logger object.
    """
    filename = "./logs/"+datetime.datetime.now().strftime("%Y_%m_%d__%H_%M_%S")+".log"
    log_config = open("./configs/log.yaml", "r").read()
    log_config = yaml.safe_load(log_config)
    log_config["filename"] = filename
    logging.config.dictConfig(log_config)
    logger = logging.getLogger(__name__)
    return logger

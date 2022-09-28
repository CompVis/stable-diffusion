"""
Wrapper functions for logging
"""
import logging as log
from datetime import datetime


def logging():
    """
    Crates and returns a logger object
    Returns:

    """
    obj = log.getLogger()
    log.basicConfig(level=log.INFO)
    return obj


def format_message(msg):
    """
    Adds timestamp to log message.
    :param msg: Message to be logged
    :return: Formatted message
    """
    return f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} {msg}"


def info(msg):
    """
    Log formatted info message.
    :param msg: Message to be logged
    :return: None
    """
    try:
        QtCore.qDebug(msg)
    except NameError:
        pass
    logging().info(format_message(msg))


def warning(msg):
    """
    Log formatted warning message.
    :param msg: Message to be logged
    :return: None
    """
    try:
        QtCore.qWarning(msg)
    except NameError:
        pass
    logging().warning(format_message(msg))


def error(msg):
    """
    Log formatted error message.
    :param msg: Message to be logged
    :return: None
    """
    try:
        QtCore.qCritical(msg)
    except NameError:
        pass
    logging().error(format_message(msg))

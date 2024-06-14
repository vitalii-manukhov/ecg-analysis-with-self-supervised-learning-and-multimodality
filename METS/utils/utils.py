"""
-*- coding = utf-8 -*-
@File : utils.py
@Software : vscode
"""
import datetime
import logging
import os
import random
import re

import numpy as np
import torch


def epoch_log(info):
    """
    Description:
        A method that logs the epoch information.

    Args:
        info: The information to be logged.

    Returns:
        None
    """
    logging.info("Start A New Epoch" + "\n" + "==========" * 8)
    logging.info(str(info) + "\n")


def init_log():
    """
    Description:
        A method that initializes the logger.

    Args:
        None

    Returns:
        None
    """
    LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
    nowtime = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    if not os.path.exists("logs/"):
        os.makedirs("logs/")

    # Initialize logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # Initialize file and console handler for logging to file and console
    file_handler = logging.FileHandler(f'logs/{nowtime}.log')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(LOG_FORMAT))

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(logging.Formatter(LOG_FORMAT))

    # Add handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)


def set_seed(seed):
    """
    Description:
        A method that sets the seed for the random number generator.

    Args:
        seed: The seed value.

    Returns:
        None
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def extract_loss_from_filename(filename):
    """
    Description:
        A method that extracts the loss value from the filename.

    Args:
        filename: The filename.

    Returns:
        float: The loss value.
    """
    match = re.search(r'loss_([0-9.]+)\.pt', filename)
    return float(match.group(1)) if match else None


def keep_top_files(folder_path, top_x):
    """
    Description:
        A method that keeps the top x files in the folder.

    Args:
        folder_path: The folder path.
        top_x: The number of top files to keep.

    Returns:
        None
    """
    # Get the list of files in the folder
    files = os.listdir(folder_path)

    # Extract the loss values from the filenames
    files_with_loss = [(file, extract_loss_from_filename(file)) for file in files]

    # Sort the files by loss value in descending order
    files_sorted = sorted(files_with_loss, key=lambda x: x[1], reverse=True)

    # Delete the top x files
    for file, _ in files_sorted[top_x:]:
        os.remove(os.path.join(folder_path, file))
        logging.warning(f"Delete File: {file}")


def get_smallest_loss_model_path(folder_path):
    """
    Description:
        A method that returns the path of the smallest loss model in the folder.

    Args:
        folder_path: The folder path.

    Returns:
        str: The path of the smallest loss model.
    """
    # Get the list of files in the folder
    files = os.listdir(folder_path)

    # Extract the loss values from the filenames
    files_with_loss = [(file, extract_loss_from_filename(file)) for file in files]
    # Filter out files without valid loss values
    files_with_loss = [(filename, loss) for filename, loss in files_with_loss if loss is not None]
    # Sort the files by loss value in ascending order
    files_sorted = sorted(files_with_loss, key=lambda x: x[1])

    # Get the path of the smallest loss model
    smallest_loss_file = files_sorted[0][0] if files_sorted else None

    # Return the path of the smallest loss model
    return os.path.join(folder_path, smallest_loss_file) if smallest_loss_file else None

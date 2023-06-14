"""
This module contains all the constants used in the project.

@Date: June 6th, 2023
@Author: Yassine RODANI
"""

from pathlib import Path

############ HOPSWORKS CONSTANTS ############

FEATURE_GROUP_VERSION = 1 #version of feature group to use

############ PATHS ############

PATH_SENTENCES = Path.cwd() / "models/sentences"
PATH_EMBEDDINGS = Path.cwd() / "models/embeddings"

PATH_DATA_BASE = Path.cwd() / "data"
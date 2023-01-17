import os


class Environment:
    FILE_PATH = os.getenv("HOMEVISION_DATA_PATH", "./data")

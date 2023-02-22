import os
import json
import pandas as pd


class FileDirHandler:
    @staticmethod
    def read_directories(directory):
        try:
            dirs = os.listdir(directory)
            return dirs
        except:
            return None

    @staticmethod
    def is_dir_exists_in_parent_dir(parent_dir, directory):
        try:
            dirs = os.listdir(parent_dir)
            for d in dirs:
                if d == directory:
                    return True
            return False
        except:
            return False

    @staticmethod
    def is_dir_exists(directory):
        try:
            dirs = os.listdir(directory)
            return True
        except:
            return False

    @staticmethod
    def read_json_file(path):
        try:
            with open(path) as jsonFile:
                json_object = json.load(jsonFile)
                jsonFile.close()
                return json_object
        except FileExistsError:
            return None
        except NotADirectoryError:
            return None

    @staticmethod
    def read_csv_file(path):
        return pd.read_csv(path, lineterminator='\n')

    @staticmethod
    def write_csv_file(path, df, index=False):
        df.to_csv(path, index=index)

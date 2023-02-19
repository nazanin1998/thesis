from lib.preprocessing.str_processing import StrProcessing
import ast


class StrProcessingImpl(StrProcessing):
    @staticmethod
    def convert_str_to_array(sample_str, max_len=64):
        my_array = ast.literal_eval(sample_str)
        my_array = my_array[:64]
        return my_array

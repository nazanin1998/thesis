from datasets import Dataset, DatasetDict


class BasicPreprocessing:
    @staticmethod
    def convert_df_to_ds(df):
        return Dataset.from_pandas(df)

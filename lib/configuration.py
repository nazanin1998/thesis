from dotenv import load_dotenv


class Configuration:
    def config(self):
        self.__config_env()

    @staticmethod
    def __config_env():
        load_dotenv()

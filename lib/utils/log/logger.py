import sys
import logging

# from rainbow_logging_handler import RainbowLoggingHandler
from datetime import datetime

"""Print a string `s` indented with `n` tabs at each newline"""


def log_line():
    print("######################################################")


def log_start_phase(phase_num, description):
    print(f"<< PHASE - {phase_num} <==> {description} @ {datetime.now()}>>")


def log_end_phase(phase_num, description):
    print(f"<< PHASE - {phase_num} <==> {description} DONE @ {datetime.now()}>> ")


def log_phase_desc(description):
    print(f"\t{description}")


def print_indented(n, s):
    index = 0
    for x in s.split('\n'):
        if index == 0:
            print(x)
        else:
            print('\t' * n + x)
        index += 1


def print_indented_key_value(key, value, intend_num=1):
    print(key, end='')
    print_indented(intend_num, f'{value}')


def set_logger():
    # setup `logging` module
    logger = logging.getLogger('test_logging')
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter("[%(asctime)s] %(name)s %(funcName)s():%(lineno)d\t%(message)s")  # same as default

    # setup `RainbowLoggingHandler`
    handler = RainbowLoggingHandler(sys.stderr, color_funcName=('black', 'yellow', True))
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    logger.debug("debug msg")
    logger.info("info msg")
    logger.warn("warn msg")
    logger.error("error msg")
    logger.critical("critical msg")

    try:
        raise RuntimeError("Opa!")
    except Exception as e:
        logger.exception(e)

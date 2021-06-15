import logging


def create_logger(logger_name):
    logger = logging.getLogger(logger_name)  # should be __name__ when inside a module
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter("%(name)s:%(message)s")
    if not logger.handlers:
        file_handler = logging.FileHandler(f"logs/{logger_name}.log")
        file_handler.setFormatter(formatter)

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)

        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
    return logger

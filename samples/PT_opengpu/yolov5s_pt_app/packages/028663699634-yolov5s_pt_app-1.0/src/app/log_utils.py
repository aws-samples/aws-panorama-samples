import logging, multiprocessing

def get_logger():
    logger = multiprocessing.get_logger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(\
                                  '[%(asctime)s| %(levelname)s| %(processName)s]| %(filename)s:%(lineno)s| %(message)s')
    handler = logging.FileHandler('/opt/aws/panorama/logs/my_app.log')
    handler.setFormatter(formatter)

    # this bit will make sure you won't have
    # duplicated messages in the output
    if not len(logger.handlers):
        logger.addHandler(handler)
    return logger

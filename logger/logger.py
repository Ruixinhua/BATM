import logging
import logging.config


def setup_logging(save_dir):
    """
    Setup logging configuration
    """
    log_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "simple": {"format": "%(message)s"},
            "datetime": {"format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"}
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": "DEBUG",
                "formatter": "simple",
                "stream": "ext://sys.stdout"
            },
            "info_file_handler": {
                "class": "logging.handlers.RotatingFileHandler",
                "level": "INFO",
                "formatter": "datetime",
                "filename": "info.log",
                "maxBytes": 10485760,
                "backupCount": 20, "encoding": "utf8"
            }
        },
        "root": {
            "level": "INFO",
            "handlers": [
                "console",
                "info_file_handler"
            ]
        }
    }
    # modify logging paths based on run config
    for _, handler in log_config['handlers'].items():
        if 'filename' in handler:
            handler['filename'] = str(save_dir / handler['filename'])

    logging.config.dictConfig(log_config)

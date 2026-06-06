import logging
import colorlog


def _build_logger() -> logging.Logger:
    handler = colorlog.StreamHandler()
    handler.setFormatter(
        colorlog.ColoredFormatter(
            fmt="%(log_color)s[%(levelname)-8s]%(reset)s %(cyan)s%(asctime)s%(reset)s  %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            log_colors={
                "DEBUG": "white",
                "INFO": "green",
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "bold_red",
            },
            reset=True,
        )
    )

    log = logging.getLogger("agentic")
    log.addHandler(handler)
    log.setLevel(logging.DEBUG)
    log.propagate = False
    return log


logger = _build_logger()

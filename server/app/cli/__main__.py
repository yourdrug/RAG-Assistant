"""Allow running CLI as `python -m cli` from the app directory."""

import logging.config

from infrastructure.logging import logging_config

from cli.cli import cli

if __name__ == "__main__":
    logging.config.dictConfig(logging_config)
    cli.execute_command()

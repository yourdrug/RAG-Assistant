"""Allow running CLI as `python -m cli` from the app directory."""

from cli.cli import cli

if __name__ == "__main__":
    cli.execute_command()

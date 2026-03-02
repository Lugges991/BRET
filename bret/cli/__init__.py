"""
Command-line interface module.
"""


def main():
    from bret.cli.main import main as _main
    return _main()


__all__ = ["main"]

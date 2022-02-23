import click
import builtins
from pathlib import Path
from rich import pretty, traceback
from rich import print

# Enable rich in entire project
pretty.install()
traceback.install(show_locals=False, suppress=[click])
builtins.print = print

# set src and project folder paths
ROOT_DIR = Path(__file__).parent
PROJ_DIR = ROOT_DIR.parent
CNF_PATH = ROOT_DIR/"configs/default.yaml"
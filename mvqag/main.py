import click
from pyfiglet import Figlet
from mvqag import PROJ_DIR, ROOT_DIR
from mvqag.utils import load_yaml

# Load project info
CONTEXT_SETTINGS = dict(auto_envvar_prefix="COMPLEX")
# Load Config file
CNF = load_yaml(ROOT_DIR/"configs/default.yaml")


# ----------------------------------------> CLI ::
# * Entry point
@click.group(invoke_without_command=True, context_settings=CONTEXT_SETTINGS)
@click.pass_context
def cli(ctx):
    if ctx.invoked_subcommand is None:
        # * BANNER
        # Find more fonts here: http://www.figlet.org/examples.html
        f = Figlet(font="fourtops")
        click.echo()
        banner = "M V Q A G"
        # banner = f"..._ {banner} _..."
        click.secho(f"{f.renderText(banner)}", fg="yellow")
        print(
            """Medical Visual Question Answering and Generation (MVQAG) CLI
    Type `[yellow]mvqag --help[/yellow]` for usage details
    """
        )
        print(ctx.get_help())

    else:
        click.secho(f"\n[@ {ctx.invoked_subcommand}] begin:", fg="cyan")
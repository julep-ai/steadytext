# AIDEV-NOTE: Entry point for python -m steadytext
# Currently only supports the download subcommand

from .download import main

if __name__ == "__main__":
    # For now, always run download
    # In the future, could add subcommands like:
    # python -m steadytext download
    # python -m steadytext generate "prompt"
    # python -m steadytext embed "text"
    main()

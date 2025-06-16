#!/usr/bin/env python3
"""
Examples of building CLI tools with SteadyText
"""

import sys
import click
import steadytext


# Simple CLI commands that always return the same output
def motivate():
    """Get a motivational quote"""
    return steadytext.generate("Write an inspiring motivational quote")


def excuse():
    """Generate an excuse for being late"""
    return steadytext.generate("Creative excuse for being late to a meeting")


def explain_error(error_code):
    """Convert error codes to user-friendly messages"""
    return steadytext.generate(f"Explain error {error_code} in simple, friendly terms")


# Command helper functions
def git_command(task):
    """Generate git commands for common tasks"""
    prompt = f"Git command to {task}. Return only the command, no explanation."
    return steadytext.generate(prompt).strip()


def sql_query(description):
    """Generate SQL queries from descriptions"""
    prompt = f"SQL query to {description}. Return only the query."
    return steadytext.generate(prompt).strip()


# Click-based CLI
@click.group()
def cli():
    """SteadyText CLI tool examples"""
    pass


@cli.command()
def quote():
    """Get a motivational quote"""
    print(motivate())


@cli.command()
def late():
    """Generate an excuse for being late"""
    print(excuse())


@cli.command()
@click.argument("error")
def error(error):
    """Explain an error code"""
    print(explain_error(error))


@cli.command()
@click.argument("task")
def git(task):
    """Generate a git command"""
    print(git_command(task))


if __name__ == "__main__":
    # If called with arguments, use click
    if len(sys.argv) > 1:
        cli()
    else:
        # Demo mode
        print("=== CLI Tool Examples ===\n")

        print("Motivational quote:")
        print(motivate()[:100] + "...\n")

        print("Excuse for being late:")
        print(excuse()[:100] + "...\n")

        print("Error explanation for ECONNREFUSED:")
        print(explain_error("ECONNREFUSED")[:100] + "...\n")

        print("Git command to undo last commit:")
        print(git_command("undo last commit but keep changes")[:50] + "...\n")

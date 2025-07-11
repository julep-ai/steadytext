#!/usr/bin/env python3
"""
Setup script for pg_steadytext Python modules
AIDEV-NOTE: This installs the Python modules into PostgreSQL's Python path
"""

from setuptools import setup
import os
import subprocess


def get_pg_python_path():
    """Get PostgreSQL's Python path for plpython3u"""
    try:
        # Try to get PostgreSQL's Python path
        result = subprocess.run(
            ["pg_config", "--pkglibdir"], capture_output=True, text=True, check=True
        )
        pg_lib_dir = result.stdout.strip()

        # Check for plpython3u library
        for py_version in ["3.11", "3.10", "3.9", "3.8"]:
            py_dir = os.path.join(pg_lib_dir, f"python{py_version}")
            if os.path.exists(py_dir):
                return py_dir

        # Fallback to site-packages
        return None
    except Exception:
        return None


# Package metadata
setup(
    name="pg_steadytext",
    version="1.2.0",
    description="PostgreSQL extension for SteadyText - Python modules",
    author="Julep AI",
    author_email="noreply@julep.ai",
    url="https://github.com/julep-ai/steadytext",
    license="PostgreSQL",
    # Python packages
    packages=["pg_steadytext"],
    package_dir={"pg_steadytext": "python"},
    # Dependencies
    install_requires=[
        "steadytext>=2.1.0",
        "pyzmq>=22.0.0",
        "numpy>=1.20.0",
    ],
    # Python version requirement
    python_requires=">=3.8",
    # Additional metadata
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: PostgreSQL License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: SQL",
        "Topic :: Database",
    ],
)

# Post-installation message
if __name__ == "__main__":
    pg_path = get_pg_python_path()
    if pg_path:
        print(f"\nDetected PostgreSQL Python path: {pg_path}")
        print("You may need to install with: pip install -e . --target " + pg_path)
    else:
        print("\nCould not detect PostgreSQL Python path.")
        print("You may need to manually install to PostgreSQL's Python directory.")

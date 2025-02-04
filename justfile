set dotenv-load := true

TEST_PYPI_TOKEN := env_var('TEST_PYPI_TOKEN')
PYPI_TOKEN := env_var('PYPI_TOKEN')
python := justfile_directory() / ".venv" / "bin" / "python"

# List all available recipes
default:
  just --list
  @echo "To execute a recipe: just [recipe-name]"

# Format with Black
black: check-uv
  uv run black {{justfile_directory()}}/src
  uv run black {{justfile_directory()}}/tests

# Run `uv build`
build *BUILD_ARGS: check-uv
  uv build --project {{justfile_directory()}} {{BUILD_ARGS}}

# Check dependencies are installed
check-dependencies: check-uv
  which pytest
  which black

# Check uv is installed
check-uv:
  @which uv

# Publish the package to PyPI
publish-pypi: check-uv
  uv publish --project {{justfile_directory()}} --token {{PYPI_TOKEN}}

# Publish the package to TestPyPI
publish-testpypi: check-uv
  uv publish --publish-url https://test.pypi.org/legacy/ --project {{justfile_directory()}} --token {{TEST_PYPI_TOKEN}}

# Run tests with PyTest
test *TEST_ARGS: check-uv
  uv run pytest {{TEST_ARGS}}

# # Test that the package can be installed and imported with `uv run`
# test-package-run: check-uv
#   uv run --with pyssed --no-project -- python -c "import pyssed"
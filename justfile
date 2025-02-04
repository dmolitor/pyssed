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

# Check uv is installed
check-uv:
  @which uv

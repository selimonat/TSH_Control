#setup:	@ set up all requirements
setup.env: setup.pip setup.poetry

#setup.pip:	@ set up pip and poetry
setup.pip:
	pip install --upgrade pip
	pip install poetry

#setup.poetry:	@ set up poetry dependencies
setup.poetry:
	poetry config virtualenvs.create true
	poetry config virtualenvs.in-project true
	poetry install --no-interaction --no-ansi

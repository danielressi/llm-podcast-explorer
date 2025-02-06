# llm-podcast-explorer

[![Release](https://img.shields.io/github/v/release/danielressi/llm-podcast-explorer)](https://img.shields.io/github/v/release/danielressi/llm-podcast-explorer)
[![Build status](https://img.shields.io/github/actions/workflow/status/danielressi/llm-podcast-explorer/main.yml?branch=main)](https://github.com/danielressi/llm-podcast-explorer/actions/workflows/main.yml?query=branch%3Amain)
[![codecov](https://codecov.io/gh/danielressi/llm-podcast-explorer/branch/main/graph/badge.svg)](https://codecov.io/gh/danielressi/llm-podcast-explorer)
[![Commit activity](https://img.shields.io/github/commit-activity/m/danielressi/llm-podcast-explorer)](https://img.shields.io/github/commit-activity/m/danielressi/llm-podcast-explorer)
[![License](https://img.shields.io/github/license/danielressi/llm-podcast-explorer)](https://img.shields.io/github/license/danielressi/llm-podcast-explorer)

This repository allows you to explore your favourite podcast in an interactive visualisation by leveraging the automated analysis of an llm workflow.

- **Github repository**: <https://github.com/danielressi/llm-podcast-explorer/>
- **Documentation** <https://danielressi.github.io/llm-podcast-explorer/>

## Getting started with your project

### 1. Create a New Repository

First, create a repository on GitHub with the same name as this project, and then run the following commands:

```bash
git init -b main
git add .
git commit -m "init commit"
git remote add origin git@github.com:danielressi/llm-podcast-explorer.git
git push -u origin main
```

### 2. Set Up Your Development Environment

Then, install the environment and the pre-commit hooks with

```bash
make install
```

This will also generate your `uv.lock` file

### 3. Run the pre-commit hooks

Initially, the CI/CD pipeline might be failing due to formatting issues. To resolve those run:

```bash
uv run pre-commit run -a
```

### 4. Commit the changes

Lastly, commit the changes made by the two steps above to your repository.

```bash
git add .
git commit -m 'Fix formatting issues'
git push origin main
```

You are now ready to start development on your project!
The CI/CD pipeline will be triggered when you open a pull request, merge to main, or when you create a new release.

To finalize the set-up for publishing to PyPI, see [here](https://fpgmaas.github.io/cookiecutter-uv/features/publishing/#set-up-for-pypi).
For activating the automatic documentation with MkDocs, see [here](https://fpgmaas.github.io/cookiecutter-uv/features/mkdocs/#enabling-the-documentation-on-github).
To enable the code coverage reports, see [here](https://fpgmaas.github.io/cookiecutter-uv/features/codecov/).

## Releasing a new version



---

Repository initiated with [fpgmaas/cookiecutter-uv](https://github.com/fpgmaas/cookiecutter-uv).

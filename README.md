# PrefSampling

[![PyPI Status](https://img.shields.io/pypi/v/prefsampling.svg)](https://pypi.python.org/pypi/prefsampling)
[![Build badge](https://github.com/COMSOC-Community/prefsampling/workflows/build/badge.svg)](https://github.com/COMSOC-Community/prefsampling/actions/workflows/build.yml)
[![codecov](https://codecov.io/gh/COMSOC-Community/prefsampling/branch/main/graphs/badge.svg)](https://codecov.io/gh/COMSOC-Community/prefsampling/tree/main)

## Overview

PrefSampling is a lightweight Python library that provides preference samplers.
These are algorithms that generate random preferences based on precisely
defined statistical cultures. We consider different type of preferences:

- Ordinal: preferences are expressed as rankings of the candidates;
- Approval: preferences are expressed by indicating a set of approved candidates.

## Installation

The package can be installed [from PyPI](https://pypi.org/project/prefsampling/) using:
```shell
pip3 install prefsampling
```

## Documentation

The complete documentation is available [here](https://comsoc-community.github.io/prefsampling/).

## Development

### Setting up the development mode

We are more than happy to receive help with the development of the package.
If you want to contribute, here are some elements to take into account.

First, install the development dependencies by running the following command:
```shell
pip install -e ".[dev]"
```

### Conventions

We try to enforce uniformity within the package. Here are some general guidelines.

- All samplers have `num_voters` and `num_candidates` as their two first positional arguments
- All samplers accept a `seed` parameter to set the seed of the random number generator

Within the package, the samplers are organised in modules based on the ballot format they
generate. The `prefsampling.core` module is used for features used across samplers.
Within the submodule corresponding to the ballot format, there is a Python file 
for each family of samplers. All the samplers are imported and appear in the `__all__`
variable of the `__init__.py` file of the corresponding module (defined byt the ballot
format).

### Tests

The tests are run with unittest. Simply run the following command to launch the tests:
```shell
python -m unittest
```

Several tests are automatised. When a new sampler is added to the package, it needs
to be added in several places for the tests. The following list provides the details:

- Add the sampler to the list `ALL_SAMPLERS` in `test_all_samplers.py`. Follow the convention for the import statements (that you can guess from the ones already there) to avoid duplicated names. The basic requirements (parameters, validation, etc.) that any sampler need to satisfy will then be checked.
- Add the sampler to corresponding test file for its ballot format: `test_all_ballotformat_samplers.py`.
- If needed, add a file `test_ballotformat_samplername.py` for tests that are specific to the sampler.

### Validation

We aim at statistically validating the samplers we provide. All the code necessary to 
run the validation is gathered in the `validation` folder of the repository.

When a new sampler is added to the package, proceed as follows:
- Create the corresponding file in the `validation/ballotformat/` folder.
- In this file, define a class that inherit from the `validation.validator.Validator`. This requires you to define a set of methods used to compute the theoretical probabilities of the outcomes of the samplers.
- Add the validator in the `run.py` file.
- Run the `run.py` file (you may want to comment out some parts).
- Copy the generated graphs in the correct place of the `doc-source/source/validation_plots` folder.
- Update the `doc-source/source/validation.rst` file accordingly.

### Documentation

The doc is generated using sphinx. We use the [numpy style guide](https://numpydoc.readthedocs.io/en/latest/format.html).
The [napoleon](https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html) extension for Sphinx is used
and the HTML style is defined by the [Book Sphinx Theme](https://sphinx-book-theme.readthedocs.io/en/stable/).

To generate the doc, first move inside the `docs-source` folder and run the following:
```shell
make clean 
make html
```

This will generate the documentation locally (in the folder `docs-source/build`). If you want the documentation 
to also be updated when pushing, run:
```shell
make github
```

After having pushed, the documentation will automatically be updated. Note that the
`github` directive may not work on Windows.

### Publishing on PyPI

The pipeline between GitHub and PyPI is automatised. To push a new version do the following:
- Update the `pyproject.toml` with the new version number.
- On GitHub, create a new release tagged with the bew version number.
- You're done, the new version of the package is automatically pushed to PyPI after the creation of a GitHub release.

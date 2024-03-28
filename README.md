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

This package is part of the
[Guide to Numerical Experiments on Elections in Computational Social Choice](https://arxiv.org/abs/2402.11765).

## Installation

The package can be installed [from PyPI](https://pypi.org/project/prefsampling/) using:
```shell
pip3 install prefsampling
```

## Documentation

The complete documentation is available [here](https://comsoc-community.github.io/prefsampling/).

## Citing our Work

If you are using this package we kindly ask you to cite the following reference to credit our work
[link](https://arxiv.org/abs/2402.11765).

```text

Boehmer N., Faliszewski P., Janeczko Ł., Kaczmarczyk A., Lisowski G., Pierczyński G., Rey S., Stolicki D., Szufa S., Wąs T. (2024).
Guide to Numerical Experiments on Elections in Computational Social Choice.
arXiv preprint arXiv:2402.11765.
```


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
variable of the `__init__.py` file of the corresponding module (defined by the ballot
format).

### Tests

The tests are run with unittest. Simply run the following command to launch the tests:
```shell
python -m unittest
```

The structure of the test module follows that of the package. There is one submodule per
ballot format we sample. Within the submodule, there is one file per statistical culture.

At the submodule level, there is a file `test_all_ballotformat_samplers.py` that gathers the
test that are common to all samplers of the given ballot format.

In the file corresponding the statistical culture, there is a function that returns all 
the samplers (with their arguments set) that are used as test cases, together with
the tests that are specific to the sampler.

When a new sampler is added to the package, it needs to be added in several places within the test
module:

- A file `test/ballotformat/test_ballotformat_culturename.py` defining the tests specific to the sampler and the functions to use for the tests (called `random_ballotformat_culturename_samplers`).
- In `test_all_ballotformat_samplers.py`, add the functions for the sampler to the `random_ballotformat_samplers()` function.
- If it is a sampler for actual ballots (i.e., not points in space or trees), add the functions for the samplers to the `random_samplers()` in the file `test/test_all_samplers.py`.

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
make githubclean
make github
```

After having pushed, the documentation will automatically be updated. Note that the
`github` directive may not work on Windows.

### Publishing on PyPI

The pipeline between GitHub and PyPI is automatised. To push a new version do the following:
- Update the `pyproject.toml` with the new version number.
- Update the `prefsampling/__init__.py` with the new version number.
- On GitHub, create a new release tagged with the new version number (only admins can do that).
- You're done, the new version of the package is automatically pushed to PyPI after the creation of a GitHub release.

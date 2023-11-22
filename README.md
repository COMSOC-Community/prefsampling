# PrefSampling

[![Build badge](https://github.com/simon-rey/prefsampling/workflows/build/badge.svg)](https://github.com/simon-rey/prefsampling/actions/workflows/build.yml)
[![codecov](https://codecov.io/gh/simon-rey/prefsampling/branch/main/graphs/badge.svg)](https://codecov.io/gh/simon-rey/prefsampling/tree/main)

A small package providing all the algorithms to sample preferences

Development
===========

We try to enforce uniformity within the package. Here are some general guidelines.

- All samplers have `num_agents` and `num_candidates` as their first positional arguments
- All samplers accept a `seed` parameter to set the seed of the random number generator

The tests are run with unittest. This is the procedure when adding a new sampler.

- Add the sampler to the list `ALL_SAMPLERS` in `test_samplers.py`. The basic requirements (parameters, validation, etc.) that any sampler need to satisfy will then be checked.
- Add the sampler to corresponding test file for its ballot format (e.g., `test_ordinal_samplers.py`).
- If needed, add a file `test_ballotformat_samplername.py` for tests that are specific to the sampler.


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

After having pushed, the documentation will automatically be updated.

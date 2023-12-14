.. _validation:

Validation
==========

To ensure the correctness of the samplers provided in this package, we implemented
various tests. First we implemented typical Python tests to make sure
that the developed features are correct from a programming perspective. But more
importantly, we also ensure that the samplers actually sample what we want.

Procedure
---------

To validate each sampler, we display the distribution of the samples it is generating
together with the theoretical distribution of these samples. This allows us to asses
the adequation between the two.

Validated Samplers
------------------

.. toctree::
    :maxdepth: 2

    ordinal
    approval
    tree

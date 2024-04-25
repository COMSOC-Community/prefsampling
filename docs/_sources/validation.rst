Validation
==========

To ensure the correctness of the samplers provided in this package, we implemented
various tests. First we implemented typical Python tests to make sure
that the developed features are correct from a programming perspective. But more
importantly, we also ensure that the samplers actually sample what we want.

To validate each sampler, we display the distribution of the samples it is generating
together with the theoretical distribution of these samples. This allows us to asses
the alignment between the two.

We try to provide validation to all samplers. It is not always possible because the theoretical
distribution is not always known. In very few cases, this shows a problem with the sampler that we
could not fix.

The validation of each sampler is presented in its reference (see for instance
:py:func:`~prefsampling.ordinal.mallows.mallows`).
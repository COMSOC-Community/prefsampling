.. _validation:

Validation
==========

In order to ensure the correctness of the samplers we provide in this package,
we implemented different tests. First we implemented typical Python tests to make sure
that the developed features are correct from a programming perspective. But more
importantly, we also implemented statistical tests to ensure that the samplers
actually sample what we want. We present these validations below.

Procedure
---------

To validate each sampler, we ran chi-squared tests to compare the distribution of the
samples it is generating to the theoretical distribution of these samples. We also
display this information in a graph to assess (non-scientifically) the adequation.

Ordinal Samplers
----------------

Impartial
~~~~~~~~~

.. image:: validation_plots/ordinal/impartial/Frequencies_Impartial.png
  :width: 500
  :alt: Observed versus theoretical frequencies for the impartial culture

Impartial Anonymous
~~~~~~~~~~~~~~~~~~~

Stratification
~~~~~~~~~~~~~~

Mallows
~~~~~~~

.. image:: validation_plots/ordinal/mallows/Frequencies_Mallows_0.5.png
  :width: 500
  :alt: Observed versus theoretical frequencies for the impartial culture

Single-Peaked
~~~~~~~~~~~~~


.. image:: validation_plots/ordinal/single_peaked/Frequencies_SP_Walsh.png
  :width: 500
  :alt: Observed versus theoretical frequencies for the impartial culture

.. image:: validation_plots/ordinal/single_peaked/Frequencies_SP_Conitzer.png
  :width: 500
  :alt: Observed versus theoretical frequencies for the impartial culture
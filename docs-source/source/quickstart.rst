.. _quickstart:

Quick Start
===========

Organisation of the package
---------------------------

The samplers are organised in the package based on the format of sample they produce.
The module :code:`prefsampling.ordinal` contains the ordinal samplers that
generate rankings of the alternative.
The module :code:`prefsampling.approval` contains the samplers for approval preferences,
the ones that generate sets of alternatives.

Sample Types
------------

To make it easy to embed the package in all kinds of tools, we use basic Python types:

* Ordinal samplers return collections of :code:`np.ndarray`, that is, `Numpy<>_` arrays where the most preferred alternative is at position 0, the next one at position 1 and so forth;
* Approval samplers return collections of :code:`set`, a set containing the approved alternatives.

General Syntax
--------------

All the sampler we provide have the same signature:

.. code-block:: python
    sampler(num_voters, num_candidates, **args, seed=None, **kwargs)

The parameter :code:`num_voters` represents the number of samples that will be generated and
the parameter :code:`num_candidates` the number of alternatives to consider.
The :code:`seed` parameter can be used to pass the seed used to defined the numpy
random number generator to give you more control if needed (for replication for instance).
Other parameters are specific to the samplers.

Samplers
--------

In the following table, we present all the samplers provided by the package, they all follow
the syntax described above.

Ordinal Samplers
~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 25 25 50
   :header-rows: 1

   * - Sampler
     - **args
     - **kwargs
   * - :py-func:`~prefsampling.ordinal.impartial.impartial`
     -
     -
   * - :py-func:`~prefsampling.ordinal.mallows.mallows`
     -
     - :code:`phi` (default to 0.5), :code:`weight` (defaults to 0)

.. _quickstart:

Quick Start
===========

Organisation of the package
---------------------------

The samplers are organised in the package based on the format of sample they produce.
The module :py:mod:`prefsampling.ordinal` contains the ordinal samplers that
generate rankings of the candidates.
The module :py:mod:`prefsampling.approval` contains the samplers for approval preferences,
the ones that generate sets of candidates.

Sample Types
------------

To make it easy to embed the package in all kinds of tools, we use basic Python types:

* Ordinal samplers return collections of :code:`np.ndarray`, that is, `Numpy <https://numpy.org/>`_ arrays where the most preferred candidate is at position 0, the next one at position 1 and so forth;
* Approval samplers return collections of :code:`set`, where each set contains the approved candidates.

In all cases, the candidates are named `0, 1, 2, ...`.

General Syntax
--------------

All the sampler we provide have the same signature:

.. code-block:: python

    sampler(num_voters, num_candidates, **args, seed=None, **kwargs)

The parameter :code:`num_voters` represents the number of samples that will be generated and
the parameter :code:`num_candidates` the number of candidates to consider.
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
     - \*\*args
     - \*\*kwargs
   * - :py:func:`~prefsampling.ordinal.identity`
     - ---
     - ---
   * - :py:func:`~prefsampling.ordinal.impartial`
     - ---
     - ---
   * - :py:func:`~prefsampling.ordinal.impartial_anonymous`
     - ---
     - ---
   * - :py:func:`~prefsampling.ordinal.mallows`
     - :code:`phi`
     - | :code:`central_vote` (defaults to `0, 1, 2, ...`)
       | :code:`normalise_phi` (defaults to :code:`False`)
   * - :py:func:`~prefsampling.ordinal.norm_mallows`
     - :code:`norm_phi`
     - | :code:`central_vote` (defaults to `0, 1, 2, ...`)
   * - :py:func:`~prefsampling.ordinal.euclidean`
     - ---
     - | :code:`space` (defaults to :py:const:`~prefsampling.core.euclidean.EuclideanSpace.UNIFORM`)
       | :code:`dimension` (defaults to 2)
   * - :py:func:`~prefsampling.ordinal.plackett_luce`
     - :code:`alphas`
     - ---
   * - :py:func:`~prefsampling.ordinal.didi`
     - :code:`alphas`
     - ---
   * - :py:func:`~prefsampling.ordinal.urn`
     - :code:`alpha`
     - ---
   * - :py:func:`~prefsampling.ordinal.stratification`
     - :code:`weight`
     - ---
   * - :py:func:`~prefsampling.ordinal.single_peaked_conitzer`
     - ---
     - ---
   * - :py:func:`~prefsampling.ordinal.single_peaked_walsh`
     - ---
     - ---
   * - :py:func:`~prefsampling.ordinal.single_peaked_circle`
     - ---
     - ---
   * - :py:func:`~prefsampling.ordinal.single_crossing`
     - ---
     - ---
   * - :py:func:`~prefsampling.ordinal.group_separable`
     - ---
     - :code:`tree_sampler` (defaults to :py:const:`~prefsampling.ordinal.TreeSampler.SCHROEDER`)


Approval Samplers
~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 25 25 50
   :header-rows: 1

   * - Sampler
     - \*\*args
     - \*\*kwargs
   * - :py:func:`~prefsampling.approval.identity`
     - :code:`p`
     - ---
   * - :py:func:`~prefsampling.approval.empty`
     - ---
     - ---
   * - :py:func:`~prefsampling.approval.full`
     - ---
     - ---
   * - :py:func:`~prefsampling.approval.impartial`
     - :code:`p`
     - ---
   * - :py:func:`~prefsampling.approval.resampling`
     - | :code:`p`
       | :code:`phi`
     - :code:`central_vote` (defaults to `{0, 1, 2, ...}`)
   * - :py:func:`~prefsampling.approval.disjoint_resampling`
     - | :code:`p`
       | :code:`phi`
     - :code:`g` (defaults to 2)
   * - :py:func:`~prefsampling.approval.moving_resampling`
     - | :code:`p`
       | :code:`phi`
     - :code:`num_legs` (defaults to 1)
   * - :py:func:`~prefsampling.approval.euclidean`
     - ---
     - | :code:`radius` (defaults to 0.5)
       | :code:`space` (defaults to :py:const:`~prefsampling.core.euclidean.EuclideanSpace.UNIFORM`)
       | :code:`dimension` (defaults to 2)
   * - :py:func:`~prefsampling.approval.noise`
     - | :code:`p`
       | :code:`phi`
     - :code:`noise_type` (defaults to :py:const:`~prefsampling.approval.NoiseType.HAMMING`)


Composition of Samplers
~~~~~~~~~~~~~~~~~~~~~~~

It is often useful to be able to compose samplers, to define mixture for instance. The functions
:py:func:`~prefsampling.core.mixture` and :py:func:`~prefsampling.core.concatenation` can do that.

The mixture uses different samplers, each being use with a given probability.

.. code-block:: python

    from prefsampling.core import mixture
    from prefsampling.ordinal import single_peaked_conitzer, single_peaked_walsh, norm_mallows

    # We create a mixture for 100 voters and 10 candidates of the single-peaked samplers using the
    # Conitzer one with probability 0.6 and the Walsh one with probability 0.4
    mixture(
        100,  # num_voters
        10,  # num_candidates
        [single_peaked_conitzer, single_peaked_walsh],  # list of samplers
        [0.6, 0.4],  # weights of the samplers
        [{}, {}]  # parameters of the samplers
    )

    # We create a mixture for 100 voters and 10 candidates of different Mallows' models
    mixture(
        100,  # num_voters
        10,  # num_candidates
        [norm_mallows, norm_mallows, norm_mallows],  # list of samplers
        [4, 10, 3],  # weights of the samplers
        [{"norm_phi": 0.4}, {"norm_phi": 0.9}, {"norm_phi": 0.23}]  # parameters of the samplers
    )

The concatenation simply concatenates the votes returned by different samplers.

.. code-block:: python

    from prefsampling.core import concatenation
    from prefsampling.ordinal import single_peaked_conitzer, single_peaked_walsh

    # We create a concatenation for 100 voters and 10 candidates. 60 votes are sampled from the
    # single_peaked_conitzer sampler and 40 votes from the single_peaked_walsh sampler.
    concatenation(
        [60, 40],  # num_voters per sampler
        10,  # num_candidates
        [single_peaked_conitzer, single_peaked_walsh],  # list of samplers
        [{}, {}]  # parameters of the samplers
    )
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

* A collection of votes represented as a :code:`list`, each element representing a voters;
* Ordinal samplers return lists of :code:`list` where the most preferred candidate is at position 0, the next one at position 1 and so forth;
* Weak ordinal samplers return lists of :code:`list` of :code:`list`, that is, indifference classes are represented as (sorted) lists, the rest follows the representation of strit ordinal votes;
* Approval samplers return lists of :code:`set`, where each set contains the approved candidates.

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

In the code reference of each sampler we provide examples of how to use the samplers in
your code. We refer the reader to these examples and do not provide a general "how to"
section here.

Ordinal Samplers
----------------

Reference: :py:mod:`prefsampling.ordinal`

.. list-table::
   :widths: 25 25 50
   :header-rows: 1

   * - Sampler
     - \*\*args
     - \*\*kwargs
   * - :py:func:`~prefsampling.ordinal.identity.identity`
     - ---
     - ---
   * - :py:func:`~prefsampling.ordinal.impartial.impartial`
     - ---
     - ---
   * - :py:func:`~prefsampling.ordinal.impartial.impartial_anonymous`
     - ---
     - ---
   * - :py:func:`~prefsampling.ordinal.mallows.mallows`
     - :code:`phi`
     - | :code:`central_vote` (defaults to `0, 1, 2, ...`)
       | :code:`normalise_phi` (defaults to :code:`False`)
       | :code:`impartial_central_vote` (defaults to :code:`False`)
   * - :py:func:`~prefsampling.ordinal.mallows.norm_mallows`
     - :code:`norm_phi`
     - | :code:`central_vote` (defaults to `0, 1, 2, ...`)
   * - :py:func:`~prefsampling.ordinal.euclidean.euclidean`
     - | :code:`num_dimensions`
       | :code:`voters_positions`
       | :code:`candidates_positions`
     - | :code:`voters_positions_args` (defaults to :code:`dict()`)
       | :code:`candidates_positions_args` (defaults to :code:`dict()`)
       | :code:`tie_radius` (defaults to :code:`None`)
   * - :py:func:`~prefsampling.ordinal.plackettluce.plackett_luce`
     - :code:`alphas`
     - ---
   * - :py:func:`~prefsampling.ordinal.didi.didi`
     - :code:`alphas`
     - ---
   * - :py:func:`~prefsampling.ordinal.urn.urn`
     - :code:`alpha`
     - ---
   * - :py:func:`~prefsampling.ordinal.impartial.stratification`
     - :code:`weight`
     - ---
   * - :py:func:`~prefsampling.ordinal.singlepeaked.single_peaked_conitzer`
     - ---
     - ---
   * - :py:func:`~prefsampling.ordinal.singlepeaked.single_peaked_walsh`
     - ---
     - ---
   * - :py:func:`~prefsampling.ordinal.singlepeaked.single_peaked_circle`
     - ---
     - ---
   * - :py:func:`~prefsampling.ordinal.singlecrossing.single_crossing`
     - ---
     - ---
   * - :py:func:`~prefsampling.ordinal.groupseparable.group_separable`
     - ---
     - :code:`tree_sampler` (defaults to :py:const:`~prefsampling.ordinal.groupseparable.TreeSampler.SCHROEDER`)


Approval Samplers
-----------------

Reference: :py:mod:`prefsampling.approval`

.. list-table::
   :widths: 25 25 50
   :header-rows: 1

   * - Sampler
     - \*\*args
     - \*\*kwargs
   * - :py:func:`~prefsampling.approval.identity.identity`
     - :code:`rel_num_approvals`
     - ---
   * - :py:func:`~prefsampling.approval.identity.empty`
     - ---
     - ---
   * - :py:func:`~prefsampling.approval.identity.full`
     - ---
     - ---
   * - :py:func:`~prefsampling.approval.impartial.impartial`
     - :code:`p`
     - ---
   * - :py:func:`~prefsampling.approval.impartial.impartial_constant_size`
     - :code:`rel_num_approvals`
     - ---
   * - :py:func:`~prefsampling.approval.urn.urn`
     - | :code:`p`
       | :code:`alpha`
     - ---
   * - :py:func:`~prefsampling.approval.urn.urn_constant_size`
     - | :code:`rel_num_approvals`
       | :code:`alpha`
     - ---
   * - :py:func:`~prefsampling.approval.urn.urn_partylist`
     - :code:`alpha`
     - | :code:`parties` (required if :code:`party_votes is None`)
       | :code:`party_votes` (required if :code:`parties is None`)
   * - :py:func:`~prefsampling.approval.resampling.resampling`
     - | :code:`phi`
       | :code:`rel_size_central_vote`
     - | :code:`central_vote` (defaults to `{0, 1, 2, ...}`)
       | :code:`impartial_central_vote` (defaults to :code:`False`)
   * - :py:func:`~prefsampling.approval.resampling.disjoint_resampling`
     - | :code:`phi`
       | :code:`rel_size_central_vote`
     - | :code:`num_central_votes` (defaults to :code:`None`)
       | :code:`central_votes` (see docs for the defaults)
       | :code:`impartial_central_votes` (defaults to :code:`False`)
   * - :py:func:`~prefsampling.approval.resampling.moving_resampling`
     - | :code:`phi`
       | :code:`rel_size_central_vote`
       | :code:`num_legs`
     - | :code:`central_votes` (see docs for the defaults)
       | :code:`impartial_central_votes` (defaults to :code:`False`)
   * - :py:func:`~prefsampling.approval.euclidean.euclidean_threshold`
     - | :code:`threshold`
       | :code:`num_dimensions`
       | :code:`voters_positions`
       | :code:`candidates_positions`
     - | :code:`voters_positions_args` (defaults to :code:`dict()`)
       | :code:`candidates_positions_args` (defaults to :code:`dict()`)
   * - :py:func:`~prefsampling.approval.euclidean.euclidean_vcr`
     - | :code:`voters_radius`
       | :code:`candidates_radius`
       | :code:`num_dimensions`
       | :code:`voters_positions`
       | :code:`candidates_positions`
     - | :code:`voters_positions_args` (defaults to :code:`dict()`)
       | :code:`candidates_positions_args` (defaults to :code:`dict()`)
   * - :py:func:`~prefsampling.approval.euclidean.euclidean_constant_size`
     - | :code:`rel_num_approvals`
       | :code:`num_dimensions`
       | :code:`voters_positions`
       | :code:`candidates_positions`
     - | :code:`voters_positions_args` (defaults to :code:`dict()`)
       | :code:`candidates_positions_args` (defaults to :code:`dict()`)
   * - :py:func:`~prefsampling.approval.noise.noise`
     - | :code:`phi`
       | :code:`rel_size_central_vote`
     - | :code:`distance` (defaults to :py:const:`~prefsampling.approval.noise.SetDistance.HAMMING`)
       | :code:`central_votes` (see docs for the defaults)
       | :code:`impartial_central_votes` (defaults to :code:`False`)
   * - :py:func:`~prefsampling.approval.truncated_ordinal.truncated_ordinal`
     - | :code:`rel_num_approvals`
       | :code:`ordinal_sampler`
       | :code:`ordinal_sampler_parameters`
     -

Composition of Samplers
-----------------------

It is often useful to be able to compose samplers, to define mixture for instance. The functions
:py:func:`~prefsampling.core.composition.mixture` and :py:func:`~prefsampling.core.composition.concatenation`
can do that.

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

Filters
-------

Filters are functions that operate on collections of votes and apply some random operation to them.
These are the filters we have implemented:

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Filter
     - Effect
   * - :py:func:`~prefsampling.core.filters.permute_voters`
     - Randomly permutes the voters
   * - :py:func:`~prefsampling.core.filters.rename_candidates`
     - Randomly rename the candidates
   * - :py:func:`~prefsampling.core.filters.resample_as_central_vote`
     - Resamples the votes using them as central votes of sampler whose definition include a central vote (e.g., :py:func:`~prefsampling.ordinal.mallows` or :py:func:`~prefsampling.approval.resampling`)
   * - :py:func:`~prefsampling.core.filters.coin_flip_ties`
     - Introduce random ties in a strict ordinal ballot

Below is an example of how to use the :py:func:`~prefsampling.core.filters.resample_as_central_vote`
filter.

.. code-block:: python

    from prefsampling.core import resample_as_central_vote
    from prefsampling.ordinal import single_crossing, norm_mallows

    num_candidates = 10
    initial_votes = single_crossing(100, num_candidates)

    resample_as_central_vote(
        initial_votes,  # The votes
        norm_mallows,  # The sampler
        {"norm_phi": 0.4, "seed": 855, "num_candidates": num_candidates},  # The arguments for the sampler
    )

Constants
---------

The constants used in the package are defined with respect to their corresponding samplers, see for
instance :py:class:`~prefsampling.core.euclidean.EuclideanSpace` or
:py:class:`~prefsampling.approval.noise.SetDistance`.
They are also all gathered in the :code:`prefsampling.CONSTANTS` enumeration.

.. code-block:: python

    from prefsampling import CONSTANTS

    CONSTANTS.BALL
    CONSTANTS.SCHROEDER
    CONSTANTS.BUNKE_SHEARER

Not that :py:class:`~prefsampling.core.euclidean.EuclideanSpace` and
:py:class:`~prefsampling.CONSTANTS` are not the same enumeration so direct comparison will fail.
Indeed, :code:`CONSTANTS.BALL == EuclideanSpace.BALL` is evaluated to :code:`False`. However,
the values are the same so :code:`CONSTANTS.BALL.value == EuclideanSpace.BALL.value` is evaluated
to :code:`True`.
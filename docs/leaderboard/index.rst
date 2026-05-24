2026 Challenge Leaderboard
==========================

Live in-distribution test scores for submissions tagged
``track-1-gnn`` or ``track-2-tnn`` on the
`TopoBench <https://github.com/geometric-intelligence/TopoBench>`__
repository. The page below is fully static and is regenerated every two
days from the ``results.json`` files committed in each labeled pull
request.

Pick a task and the dataset axes (homophily, average degree, power-law
exponent) to drill into a specific GraphUniverse configuration, or leave
the axes on **All** for the overall best in-distribution score.

.. raw:: html

   <iframe src="../_static/leaderboard/index.html"
           title="TDL Challenge 2026 Leaderboard"
           loading="lazy"
           style="width: 100%; min-height: 1100px; border: 0; border-radius: 0.5rem; box-shadow: 0 1px 3px rgba(15,23,42,0.08);"></iframe>

.. note::

   To submit a model, open a pull request that adds your generated
   ``results.json`` (produced by the
   ``2026_tdl_challenge/run_evaluation.ipynb`` notebook) under the
   ``2026_tdl_challenge/`` directory and apply the appropriate label
   (``track-1-gnn`` for GNN submissions, ``track-2-tnn`` for TNN
   submissions). The leaderboard refresher will pick it up on its next
   run.

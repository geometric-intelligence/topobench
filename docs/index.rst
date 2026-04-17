🌐 TopoBench (TB) 🍩
==========================

.. figure:: https://github.com/geometric-intelligence/TopoBench/raw/main/resources/logo.jpg
   :alt: topobench
   :class: with-shadow
   :width: 1000px

`TopoBench` (TB) is a modular Python library designed to standardize benchmarking and accelerate research in Topological Deep Learning (TDL).
In particular, TB allows training and comparing the performances of all sorts of Topological Neural Networks (TNNs) across the different topological domains,
where by *topological domain* we refer to a graph, a simplicial complex, a cellular complex, or a hypergraph.

.. figure:: https://github.com/geometric-intelligence/TopoBench/raw/main/resources/workflow.jpg
   :alt: workflow
   :class: with-shadow
   :width: 1000px

:pushpin: Overview
------------------

``TopoBench`` (TB) is a modular Python library designed to standardize
benchmarking and accelerate research in Topological Deep Learning (TDL).
In particular, TB allows training and comparing the performances of all
sorts of Topological Neural Networks (TNNs) across the different
topological domains, where by *topological domain* we refer to a graph,
a simplicial complex, a cellular complex, or a hypergraph. For detailed
information, please refer to the
`TopoBench: A Framework for Benchmarking Topological Deep Learning <https://arxiv.org/pdf/2406.06642>`__
paper.

.. raw:: html

   <p align="center">

.. raw:: html

   </p>

The main pipeline trains and evaluates a wide range of state-of-the-art
TNNs and Graph Neural Networks (GNNs) (see :gear: Neural Networks) on
numerous and varied datasets and benchmark tasks (see :books: Datasets
). Additionally, the library offers the ability to transform,
i.e. *lift*, each dataset from one topological domain to another (see
:rocket: Liftings), enabling for the first time an exhaustive
inter-domain comparison of TNNs.

🧩 Get Started
--------------

Create Environment
~~~~~~~~~~~~~~~~~~

First, clone and navigate to the ``TopoBench`` repository

.. code:: bash

   git clone git@github.com:geometric-intelligence/topobench.git
   cd TopoBench

Ensure ``conda`` is installed:

.. code:: bash

   conda --version || echo "Conda not found! Please install it from https://docs.anaconda.com/free/miniconda/miniconda-install/"

Next, set up and activate a conda environment ``tb`` with Python 3.11.3:

.. code:: bash

   conda create -n tb python=3.11.3
   conda activate tb

Next, check the CUDA version of your machine:

.. code:: bash

   which nvcc && nvcc --version

and ensure that it matches the CUDA version specified in the
``env_setup.sh`` file (``CUDA=cu121`` by default). If it does not match,
update ``env_setup.sh`` accordingly by changing both the ``CUDA`` and
``TORCH`` environment variables to compatible values as specified on
`this website <https://github.com/pyg-team/pyg-lib>`__.

Next, set up the environment with the following command.

.. code:: bash

   source env_setup.sh

This command installs the ``TopoBench`` library and its dependencies.

Run Training Pipeline
~~~~~~~~~~~~~~~~~~~~~

Next, train the neural networks by running the following command:

.. code:: bash

   python -m topobench

Customizing Experiment Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Thanks to ``hydra`` implementation, one can easily override the default
experiment configuration through the command line. For instance, the
model and dataset can be selected as:

::

   python -m topobench model=cell/cwn dataset=graph/MUTAG

**Remark:** By default, our pipeline identifies the source and
destination topological domains, and applies a default lifting between
them if required.

Transforms allow you to modify your data before processing. There are
two main ways to configure transforms: individual transforms and
transform groups.


Configuring Individual Transforms
---------------------------------

When configuring a single transform, follow these steps:

1. Choose a desired transform (e.g., a lifting transform).
2. Identify the relative path to the transform configuration.

The folder structure for transforms is as follows:

.. code-block:: none

   ├── configs
   │ ├── data_manipulations
   │ ├── transforms
   │ │ └── liftings
   │ │   ├── graph2cell
   │ │   ├── graph2hypergraph
   │ │   └── graph2simplicial

To override the default transform, use the following command structure:

.. code-block:: bash

   python -m topobench model=<model_type>/<model_name> dataset=<data_type>/<dataset_name> transforms=[<transform_path>/<transform_name>]

For example, to use the ``discrete_configuration_complex`` lifting with
the ``cell/cwn`` model:

.. code-block:: bash

   python -m topobench model=cell/cwn dataset=graph/MUTAG transforms=[liftings/graph2cell/discrete_configuration_complex]

Configuring Transform Groups
----------------------------

For more complex scenarios, such as combining multiple data
manipulations, use transform groups:

1. Create a new configuration file in the ``configs/transforms``
   directory (e.g., ``custom_example.yaml``).
2. Define the transform group in the YAML file:

.. code-block:: yaml

   defaults:
   - data_manipulations@data_transform_1: identity
   - data_manipulations@data_transform_2: node_degrees
   - data_manipulations@data_transform_3: one_hot_node_degree_features
   - liftings/graph2cell@graph2cell_lifting: cycle

**Important:** When composing multiple data manipulations, use the ``@``
operator to assign unique names to each transform.

3. Run the experiment with the custom transform group:

.. code-block:: bash

   python -m topobench model=cell/cwn dataset=graph/ZINC transforms=custom_example

This approach allows you to create complex transform pipelines,
including multiple data manipulations and liftings, in a single
configuration file.

Additional Notes
----------------

-  **Automatic Lifting:** By default, our pipeline identifies the source
   and destination topological domains and applies a default lifting
   between them if required.
-  **Fine-Grained Configuration:** The same CLI override mechanism
   applies when modifying finer configurations within a
   ``CONFIG GROUP``.
   Please refer to the official `hydra documentation <https://hydra.cc/docs/intro/>`__
   for further details.

By mastering these configuration options, you can easily customize your experiments to suit your specific needs, from simple model and dataset selections to complex data transformation pipelines.


Experiments Reproducibility
~~~~~~~~~~~~~~~~~~~~~~~~~~~

To reproduce Table 1 from the
`TopoBench: A Framework for Benchmarking Topological Deep Learning <https://arxiv.org/pdf/2406.06642>`__
paper, please run the following command:

.. code:: bash

   bash scripts/reproduce.sh

**Remark:** We have additionally provided a public `W&B (Weights &
Biases)
project <https://wandb.ai/telyatnikov_sap/TopoBenchmark_main?nw=nwusertelyatnikov_sap>`__
with logs for the corresponding runs (updated on June 11, 2024).

Tutorials
~~~~~~~~~~~~~~~~~~

Explore our
`tutorials <https://github.com/geometric-intelligence/TopoBench/tree/main/tutorials>`__
for further details on how to add new datasets, transforms/liftings, and
benchmark tasks.

Neural Networks
~~~~~~~~~~~~~~~~~~~~~~


We list the neural networks trained and evaluated by ``TopoBench``,
organized by the topological domain over which they operate: graph,
simplicial complex, cellular complex or hypergraph. Many of these neural
networks were originally implemented in
`TopoModelX <https://github.com/pyt-team/TopoModelX>`__.


Graphs
------

+----------+----------------------------------------------------------+
| Model    | Reference                                                |
+==========+==========================================================+
| GAT      | `Graph Attention                                         |
|          | Networks <https://openreview.net/pdf?id=rJXMpikCZ>`__    |
+----------+----------------------------------------------------------+
| GIN      | `How Powerful are Graph Neural                           |
|          | Networks? <https://openreview.net/pdf?id=ryGs6iA5Km>`__  |
+----------+----------------------------------------------------------+
| GCN      | `Semi-Supervised Classification with Graph Convolutional |
|          | Networks <https://arxiv.org/pdf/1609.02907v4>`__         |
+----------+----------------------------------------------------------+
| GraphMLP | `Graph-MLP: Node Classification without Message Passing  |
|          | in Graph <https://arxiv.org/pdf/2106.04051>`__           |
+----------+----------------------------------------------------------+

Simplicial complexes
--------------------

+-----------------------------------+-----------------------------------+
| Model                             | Reference                         |
+===================================+===================================+
| SAN                               | `Simplicial Attention Neural      |
|                                   | Networks <htt                     |
|                                   | ps://arxiv.org/pdf/2203.07485>`__ |
+-----------------------------------+-----------------------------------+
| SCCN                              | `Efficient Representation         |
|                                   | Learning for Higher-Order Data    |
|                                   | with Simplicial                   |
|                                   | Complexes <https://ope            |
|                                   | nreview.net/pdf?id=nGqJY4DODN>`__ |
+-----------------------------------+-----------------------------------+
| SCCNN                             | `Convolutional Learning on        |
|                                   | Simplicial                        |
|                                   | Complexes <htt                    |
|                                   | ps://arxiv.org/pdf/2301.11163>`__ |
+-----------------------------------+-----------------------------------+
| SCN                               | `Simplicial Complex Neural        |
|                                   | Networks <https://ieeexplo        |
|                                   | re.ieee.org/document/10285604>`__ |
+-----------------------------------+-----------------------------------+

Cellular complexes
------------------

+-----------------------------------+-----------------------------------+
| Model                             | Reference                         |
+===================================+===================================+
| CAN                               | `Cell Attention                   |
|                                   | Network <htt                      |
|                                   | ps://arxiv.org/pdf/2209.08179>`__ |
+-----------------------------------+-----------------------------------+
| CCCN                              | Inspired by `A learning algorithm |
|                                   | for computational connected       |
|                                   | cellular                          |
|                                   | network <https://ieeexplo         |
|                                   | re.ieee.org/document/1202221>`__, |
|                                   | implementation adapted from       |
|                                   | `Generalized Simplicial Attention |
|                                   | Neural                            |
|                                   | Networks <htt                     |
|                                   | ps://arxiv.org/abs/2309.02138>`__ |
+-----------------------------------+-----------------------------------+
| CXN                               | `Cell Complex Neural              |
|                                   | Networks <https://open            |
|                                   | review.net/pdf?id=6Tq18ySFpGU>`__ |
+-----------------------------------+-----------------------------------+
| CWN                               | `Weisfeiler and Lehman Go         |
|                                   | Cellular: CW                      |
|                                   | Networks <htt                     |
|                                   | ps://arxiv.org/pdf/2106.12575>`__ |
+-----------------------------------+-----------------------------------+

Hypergraphs
-----------

+-----------------------------------+-----------------------------------+
| Model                             | Reference                         |
+===================================+===================================+
| AllDeepSet                        | `You are AllSet: A Multiset       |
|                                   | Function Framework for Hypergraph |
|                                   | Neural                            |
|                                   | Networks <https://open            |
|                                   | review.net/pdf?id=hpBTIv2uy_E>`__ |
+-----------------------------------+-----------------------------------+
| AllSetTransformer                 | `You are AllSet: A Multiset       |
|                                   | Function Framework for Hypergraph |
|                                   | Neural                            |
|                                   | Networks <https://open            |
|                                   | review.net/pdf?id=hpBTIv2uy_E>`__ |
+-----------------------------------+-----------------------------------+
| EDGNN                             | `Equivariant Hypergraph Diffusion |
|                                   | Neural                            |
|                                   | Operators <htt                    |
|                                   | ps://arxiv.org/pdf/2207.06680>`__ |
+-----------------------------------+-----------------------------------+
| UniGNN                            | `UniGNN: a Unified Framework for  |
|                                   | Graph and Hypergraph Neural       |
|                                   | Networks <htt                     |
|                                   | ps://arxiv.org/pdf/2105.00956>`__ |
+-----------------------------------+-----------------------------------+
| UniGNN2                           | `UniGNN: a Unified Framework for  |
|                                   | Graph and Hypergraph Neural       |
|                                   | Networks <htt                     |
|                                   | ps://arxiv.org/pdf/2105.00956>`__ |
+-----------------------------------+-----------------------------------+

Combinatorial complexes
-----------------------

+-----------------------------------+-----------------------------------+
| Model                             | Reference                         |
+===================================+===================================+
| GCCN                              | `TopoTune: A Framework for        |
|                                   | Generalized Combinatorial Complex |
|                                   | Neural                            |
|                                   | Networks <htt                     |
|                                   | ps://arxiv.org/pdf/2410.06530>`__ |
+-----------------------------------+-----------------------------------+

**Remark:** TopoBench includes
`TopoTune <https://arxiv.org/pdf/2410.06530>`__, a comprehensive
framework for easily designing new, general TDL models on any domain
using any (graph) neural network as a backbone. Please check out the
extended `TopoTune wiki
page <https://github.com/geometric-intelligence/TopoBench/wiki/TopoTune>`__
for further details on how to leverage this framework to define and
train customized topological neural network architectures.

Liftings & Transforms
~~~~~~~~~~~~~~~~~~~~~~

We list the liftings used in ``TopoBench`` to transform datasets. Here,
a *lifting* refers to a function that transforms a dataset defined on a
topological domain (*e.g.*, on a graph) into the same dataset but
supported on a different topological domain (*e.g.*, on a simplicial
complex).

Structural Liftings
-------------------



The structural lifting is responsible for the transformation of the
underlying relationships or elements of the data. For instance, it might
determine how nodes and edges in a graph are mapped into triangles and
tetrahedra in a simplicial complex. This structural transformation can
be further categorized into connectivity-based, where the mapping relies
solely on the existing connections within the data, and feature-based,
where the data’s inherent properties or features guide the new
structure.

We enumerate below the structural liftings currently implemented in
``TopoBench``; please check out the provided description links for
further details.

**Remark:**: Most of these liftings are adaptations of winner
submissions of the ICML TDL Challenge 2024
(`paper <https://proceedings.mlr.press/v251/bernardez24a.html>`__ \|
`repo <https://github.com/pyt-team/challenge-icml-2024>`__); see the
`Structural Liftings
wiki <https://github.com/geometric-intelligence/TopoBench/wiki/Structural-Liftings>`__
for a complete list of compatible liftings.

Graph to Simplicial Complex
^^^^^^^^^^^^^^^^^^^^^^^^^^^

+----------------------+----------------------+----------------------+
| Name                 | Type                 | Description          |
+======================+======================+======================+
| DnD Lifting          | Feature-based        | `Wiki                |
|                      |                      | page <https://github |
|                      |                      | .com/geometric-intel |
|                      |                      | ligence/TopoBench/wi |
|                      |                      | ki/DnD-Lifting-(Grap |
|                      |                      | h-to-Simplicial)>`__ |
+----------------------+----------------------+----------------------+
| Random Latent Clique | Connectivity-based   | `Wiki                |
| Lifting              |                      | page <https://git    |
|                      |                      | hub.com/geometric-in |
|                      |                      | telligence/TopoBench |
|                      |                      | /wiki/Random-Latent- |
|                      |                      | Clique-Lifting-(Grap |
|                      |                      | h-to-Simplicial)>`__ |
+----------------------+----------------------+----------------------+
| Line Lifting         | Connectivity-based   | `Wiki                |
|                      |                      | p                    |
|                      |                      | age <https://github. |
|                      |                      | com/geometric-intell |
|                      |                      | igence/TopoBench/wik |
|                      |                      | i/Line-Lifting-(Grap |
|                      |                      | h-to-Simplicial)>`__ |
+----------------------+----------------------+----------------------+
| Neighbourhood        | Connectivity-based   | `Wiki                |
| Complex Lifting      |                      | page <https://gith   |
|                      |                      | ub.com/geometric-int |
|                      |                      | elligence/TopoBench/ |
|                      |                      | wiki/Neighbourhood-C |
|                      |                      | omplex-Lifting-(Grap |
|                      |                      | h-to-Simplicial)>`__ |
+----------------------+----------------------+----------------------+
| Graph Induced        | Connectivity-based   | `Wiki                |
| Lifting              |                      | page <http           |
|                      |                      | s://github.com/geome |
|                      |                      | tric-intelligence/To |
|                      |                      | poBench/wiki/Graph-I |
|                      |                      | nduced-Lifting-(Grap |
|                      |                      | h-to-Simplicial)>`__ |
+----------------------+----------------------+----------------------+
| Eccentricity Lifting | Connectivity-based   | `Wiki                |
|                      |                      | page <htt            |
|                      |                      | ps://github.com/geom |
|                      |                      | etric-intelligence/T |
|                      |                      | opoBench/wiki/Eccent |
|                      |                      | ricity-Lifting-(Grap |
|                      |                      | h-to-Simplicial)>`__ |
+----------------------+----------------------+----------------------+
| Feature‐Based Rips   | Both connectivity    | `Wiki                |
| Complex              | and feature-based    | pag                  |
|                      |                      | e <https://github.co |
|                      |                      | m/geometric-intellig |
|                      |                      | ence/TopoBench/wiki/ |
|                      |                      | Feature%E2%80%90Base |
|                      |                      | d-Rips-Complex-(Grap |
|                      |                      | h-to-Simplicial)>`__ |
+----------------------+----------------------+----------------------+
| Clique Lifting       | Connectivity-based   | `Wiki                |
|                      |                      | pag                  |
|                      |                      | e <https://github.co |
|                      |                      | m/geometric-intellig |
|                      |                      | ence/TopoBench/wiki/ |
|                      |                      | Clique-Lifting-(Grap |
|                      |                      | h-to-Simplicial)>`__ |
+----------------------+----------------------+----------------------+
| K-hop Lifting        | Connectivity-based   | `Wiki                |
|                      |                      | p                    |
|                      |                      | age <https://github. |
|                      |                      | com/geometric-intell |
|                      |                      | igence/TopoBench/wik |
|                      |                      | i/KHop-Lifting-(Grap |
|                      |                      | h-to-Simplicial)>`__ |
+----------------------+----------------------+----------------------+

Graph to Cell Complex
^^^^^^^^^^^^^^^^^^^^^

+-----------------------+--------------------+-----------------------+
| Name                  | Type               | Description           |
+=======================+====================+=======================+
| Discrete              | Connectivity-based | `Wiki                 |
| Configuration Complex |                    | page <ht              |
|                       |                    | tps://github.com/geom |
|                       |                    | etric-intelligence/To |
|                       |                    | poBench/wiki/Discrete |
|                       |                    | -Configuration-Comple |
|                       |                    | x-(Graph-to-Cell)>`__ |
+-----------------------+--------------------+-----------------------+
| Cycle Lifting         | Connectivity-based | `Wiki                 |
|                       |                    | page <https:          |
|                       |                    | //github.com/geometri |
|                       |                    | c-intelligence/TopoBe |
|                       |                    | nch/wiki/Cycle-Liftin |
|                       |                    | g-(Graph-to-Cell)>`__ |
+-----------------------+--------------------+-----------------------+

Graph to Hypergraph
^^^^^^^^^^^^^^^^^^^

+-----------------------+-----------------------+-----------------------+
| Name                  | Type                  | Description           |
+=======================+=======================+=======================+
| Expander Hypergraph   | Connectivity-based    | `Wiki                 |
| Lifting               |                       | page <https           |
|                       |                       | ://github.com/geometr |
|                       |                       | ic-intelligence/TopoB |
|                       |                       | ench/wiki/Expander-Hy |
|                       |                       | pergraph-Lifting-(Gra |
|                       |                       | ph-to-Hypergraph)>`__ |
+-----------------------+-----------------------+-----------------------+
| Kernel Lifting        | Both connectivity and | `Wiki                 |
|                       | feature-based         | page <https://githu   |
|                       |                       | b.com/geometric-intel |
|                       |                       | ligence/TopoBench/wik |
|                       |                       | i/Kernel-Lifting-(Gra |
|                       |                       | ph-to-Hypergraph)>`__ |
+-----------------------+-----------------------+-----------------------+
| Mapper Lifting        | Connectivity-based    | `Wiki                 |
|                       |                       | page <https://githu   |
|                       |                       | b.com/geometric-intel |
|                       |                       | ligence/TopoBench/wik |
|                       |                       | i/Mapper-Lifting-(Gra |
|                       |                       | ph-to-Hypergraph)>`__ |
+-----------------------+-----------------------+-----------------------+
| Forman‐Ricci          | Connectivity-based    | `Wiki                 |
| Curvature Coarse      |                       | page <https://git     |
| Geometry Lifting      |                       | hub.com/geometric-int |
|                       |                       | elligence/TopoBench/w |
|                       |                       | iki/Forman%E2%80%90Ri |
|                       |                       | cci-Curvature-Coarse- |
|                       |                       | Geometry-Lifting-(Gra |
|                       |                       | ph-to-Hypergraph)>`__ |
+-----------------------+-----------------------+-----------------------+
| KNN Lifting           | Feature-based         | `Wiki                 |
|                       |                       | page <https://gi      |
|                       |                       | thub.com/geometric-in |
|                       |                       | telligence/TopoBench/ |
|                       |                       | wiki/KNN-Lifting-(Gra |
|                       |                       | ph-to-Hypergraph)>`__ |
+-----------------------+-----------------------+-----------------------+
| K-hop Lifting         | Connectivity-based    | `Wiki                 |
|                       |                       | page <https://git     |
|                       |                       | hub.com/geometric-int |
|                       |                       | elligence/TopoBench/w |
|                       |                       | iki/KHop-Lifting-(Gra |
|                       |                       | ph-to-Hypergraph)>`__ |
+-----------------------+-----------------------+-----------------------+

Pointcloud to Simplicial
^^^^^^^^^^^^^^^^^^^^^^^^^

+---------------------+---------------+------------------------------+
| Name                | Type          | Description                  |
+=====================+===============+==============================+
| Delaunay Lifting    | Feature-based | `Wiki                        |
|                     |               | page <https://github.com/g   |
|                     |               | eometric-intelligence/TopoBe |
|                     |               | nch/wiki/Delaunay-Lifting-(P |
|                     |               | ointcloud-to-Simplicial)>`__ |
+---------------------+---------------+------------------------------+
| Random Flag Complex | Feature-based | `Wiki                        |
|                     |               | p                            |
|                     |               | age <https://github.com/geom |
|                     |               | etric-intelligence/TopoBench |
|                     |               | /wiki/Random-Flag-Complex-(P |
|                     |               | ointcloud-to-Simplicial)>`__ |
+---------------------+---------------+------------------------------+

Pointcloud to Hypergraph
^^^^^^^^^^^^^^^^^^^^^^^^

+-------------------------+---------------+-------------------------+
| Name                    | Type          | Description             |
+=========================+===============+=========================+
| Mixture of Gaussians    | Feature-based | `Wiki                   |
| MST lifting             |               | page <https:/           |
|                         |               | /github.com/geometric-i |
|                         |               | ntelligence/TopoBench/w |
|                         |               | iki/Mixture-of-Gaussian |
|                         |               | s---MST-lifting-(Pointc |
|                         |               | loud-to-Hypergraph)>`__ |
+-------------------------+---------------+-------------------------+
| PointNet Lifting        | Feature-based | `Wiki                   |
|                         |               | page <https://githu     |
|                         |               | b.com/geometric-intelli |
|                         |               | gence/TopoBench/wiki/Po |
|                         |               | intNet--Lifting-(Pointc |
|                         |               | loud-to-Hypergraph)>`__ |
+-------------------------+---------------+-------------------------+
| Voronoi Lifting         | Feature-based | `Wiki                   |
|                         |               | page <https://git       |
|                         |               | hub.com/geometric-intel |
|                         |               | ligence/TopoBench/wiki/ |
|                         |               | Voronoi-Lifting-(Pointc |
|                         |               | loud-to-Hypergraph)>`__ |
+-------------------------+---------------+-------------------------+

Simplicial to Combinatorial
^^^^^^^^^^^^^^^^^^^^^^^^^^^

+----------------+--------------------+------------------------------+
| Name           | Type               | Description                  |
+================+====================+==============================+
| Coface Lifting | Connectivity-based | `Wiki                        |
|                |                    | page <https://github.com/ge  |
|                |                    | ometric-intelligence/TopoBen |
|                |                    | ch/wiki/Coface-Lifting-(Simp |
|                |                    | licial-to-Combinatorial)>`__ |
+----------------+--------------------+------------------------------+

Hypergraph to Combinatorial
^^^^^^^^^^^^^^^^^^^^^^^^^^^

+-----------------------+--------------------+-----------------------+
| Name                  | Type               | Description           |
+=======================+====================+=======================+
| Universal Strict      | Connectivity-based | `Wiki                 |
| Lifting               |                    | page <https://gi      |
|                       |                    | thub.com/geometric-in |
|                       |                    | telligence/TopoBench/ |
|                       |                    | wiki/Universal-Strict |
|                       |                    | -Lifting-(Hypergraph- |
|                       |                    | to-Combinatorial)>`__ |
+-----------------------+--------------------+-----------------------+

Feature Liftings
----------------

Feature liftings address the transfer of data attributes or features
during mapping, ensuring that the properties associated with the data
elements are consistently preserved in the new representation.

+-----------+----------------------------------------------+----------+
| Name      | Description                                  | S        |
|           |                                              | upported |
|           |                                              | Domains  |
+===========+==============================================+==========+
| Proj      | Projects r-cell features of a graph to       | All      |
| ectionSum | r+1-cell structures utilizing incidence      |          |
|           | matrices (B_{r}).                            |          |
+-----------+----------------------------------------------+----------+
| Co        | Concatenate r-cell features to obtain        | Si       |
| ncatenati | r+1-cell features.                           | mplicial |
| onLifting |                                              |          |
+-----------+----------------------------------------------+----------+

Data Transformations
~~~~~~~~~~~~~~~~~~~~

Specially useful in pre-processing steps, these are the general data
manipulations currently implemented in ``TopoBench``:

+-----------------------------------+-----------------------------------+
| Transform                         | Description                       |
+===================================+===================================+
| OneHotDegreeFeatures              | Adds the node degree as one hot   |
|                                   | encodings to the node features.   |
+-----------------------------------+-----------------------------------+
| NodeFeaturesToFloat               | Converts the node features of the |
|                                   | input graph to float.             |
+-----------------------------------+-----------------------------------+
| NodeDegrees                       | Calculates the node degrees of    |
|                                   | the input graph.                  |
+-----------------------------------+-----------------------------------+
| NodeDegrees                       | Keeps only the selected fields of |
|                                   | the input data.                   |
+-----------------------------------+-----------------------------------+
| KeepOnlyConnectedComponent        | Keep only the largest connected   |
|                                   | components of the input graph.    |
+-----------------------------------+-----------------------------------+
| InfereRadiusConnectivity          | Generates the radius connectivity |
|                                   | of the input point cloud.         |
+-----------------------------------+-----------------------------------+
| InfereKNNConnectivity             | Generates the k-nearest neighbor  |
|                                   | connectivity of the input point   |
|                                   | cloud.                            |
+-----------------------------------+-----------------------------------+
| IdentityTransform                 | An identity transform that does   |
|                                   | nothing to the input data.        |
+-----------------------------------+-----------------------------------+
| EqualGausFeatures                 | Generates equal Gaussian features |
|                                   | for all nodes.                    |
+-----------------------------------+-----------------------------------+
| CalculateSimplicialCurvature      | Calculates the simplicial         |
|                                   | curvature of the input graph.     |
+-----------------------------------+-----------------------------------+

.. raw:: html

   </details>

:books: Datasets
~~~~~~~~~~~~~~~~


Graph
-----

+-----------------+-----------------+-----------------+-----------------+
| Dataset         | Task            | Description     | Reference       |
+=================+=================+=================+=================+
| Cora            | Classification  | Cocitation      | `Source <h      |
|                 |                 | dataset.        | ttps://link.spr |
|                 |                 |                 | inger.com/artic |
|                 |                 |                 | le/10.1023/A:10 |
|                 |                 |                 | 09953814988>`__ |
+-----------------+-----------------+-----------------+-----------------+
| Citeseer        | Classification  | Cocitation      | `Source <htt    |
|                 |                 | dataset.        | ps://dl.acm.org |
|                 |                 |                 | /doi/10.1145/27 |
|                 |                 |                 | 6675.276685>`__ |
+-----------------+-----------------+-----------------+-----------------+
| Pubmed          | Classification  | Cocitation      | `Source         |
|                 |                 | dataset.        | <https://ojs.aa |
|                 |                 |                 | ai.org/aimagazi |
|                 |                 |                 | ne/index.php/ai |
|                 |                 |                 | magazine/articl |
|                 |                 |                 | e/view/2157>`__ |
+-----------------+-----------------+-----------------+-----------------+
| MUTAG           | Classification  | Graph-level     | `               |
|                 |                 | classification. | Source <https:/ |
|                 |                 |                 | /pubs.acs.org/d |
|                 |                 |                 | oi/abs/10.1021/ |
|                 |                 |                 | jm00106a046>`__ |
+-----------------+-----------------+-----------------+-----------------+
| PROTEINS        | Classification  | Graph-level     | `Source         |
|                 |                 | classification. |  <https://acade |
|                 |                 |                 | mic.oup.com/bio |
|                 |                 |                 | informatics/art |
|                 |                 |                 | icle/21/suppl_1 |
|                 |                 |                 | /i47/202991>`__ |
+-----------------+-----------------+-----------------+-----------------+
| NCI1            | Classification  | Graph-level     | `Source <htt    |
|                 |                 | classification. | ps://ieeexplore |
|                 |                 |                 | .ieee.org/docum |
|                 |                 |                 | ent/4053093>`__ |
+-----------------+-----------------+-----------------+-----------------+
| NCI109          | Classification  | Graph-level     | `Source <https: |
|                 |                 | classification. | //arxiv.org/pdf |
|                 |                 |                 | /2007.08663>`__ |
+-----------------+-----------------+-----------------+-----------------+
| IMDB-BIN        | Classification  | Graph-level     | `Source <https  |
|                 |                 | classification. | ://dl.acm.org/d |
|                 |                 |                 | oi/10.1145/2783 |
|                 |                 |                 | 258.2783417>`__ |
+-----------------+-----------------+-----------------+-----------------+
| IMDB-MUL        | Classification  | Graph-level     | `Source <https  |
|                 |                 | classification. | ://dl.acm.org/d |
|                 |                 |                 | oi/10.1145/2783 |
|                 |                 |                 | 258.2783417>`__ |
+-----------------+-----------------+-----------------+-----------------+
| REDDIT          | Classification  | Graph-level     | `Source <       |
|                 |                 | classification. | https://proceed |
|                 |                 |                 | ings.neurips.cc |
|                 |                 |                 | /paper_files/pa |
|                 |                 |                 | per/2017/file/5 |
|                 |                 |                 | dd9db5e033da9c6 |
|                 |                 |                 | fb5ba83c7a7ebea |
|                 |                 |                 | 9-Paper.pdf>`__ |
+-----------------+-----------------+-----------------+-----------------+
| Amazon          | Classification  | Heterophilic    | `Source <https  |
|                 |                 | dataset.        | ://arxiv.org/pd |
|                 |                 |                 | f/1205.6233>`__ |
+-----------------+-----------------+-----------------+-----------------+
| Minesweeper     | Classification  | Heterophilic    | `Source <https: |
|                 |                 | dataset.        | //arxiv.org/pdf |
|                 |                 |                 | /2302.11640>`__ |
+-----------------+-----------------+-----------------+-----------------+
| Empire          | Classification  | Heterophilic    | `Source <https: |
|                 |                 | dataset.        | //arxiv.org/pdf |
|                 |                 |                 | /2302.11640>`__ |
+-----------------+-----------------+-----------------+-----------------+
| Tolokers        | Classification  | Heterophilic    | `Source <https: |
|                 |                 | dataset.        | //arxiv.org/pdf |
|                 |                 |                 | /2302.11640>`__ |
+-----------------+-----------------+-----------------+-----------------+
| US-county-demos | Regression      | In turn each    | `Source <https: |
|                 |                 | node attribute  | //arxiv.org/pdf |
|                 |                 | is used as the  | /2002.08274>`__ |
|                 |                 | target label.   |                 |
+-----------------+-----------------+-----------------+-----------------+
| ZINC            | Regression      | Graph-level     | `Source <h      |
|                 |                 | regression.     | ttps://pubs.acs |
|                 |                 |                 | .org/doi/10.102 |
|                 |                 |                 | 1/ci3001277>`__ |
+-----------------+-----------------+-----------------+-----------------+

Simplicial
----------

+-----------------+-----------------+-----------------+-----------------+
| Dataset         | Task            | Description     | Reference       |
+=================+=================+=================+=================+
| Mantra          | Classification, | Predict         | `So             |
|                 | Multi-label     | topological     | urce <https://g |
|                 | Classification  | attributes of   | ithub.com/aidos |
|                 |                 | manifold        | -lab/MANTRA>`__ |
|                 |                 | triangulations  |                 |
+-----------------+-----------------+-----------------+-----------------+

Hypergraph
----------

+----------------+----------------+----------------+----------------+
| Dataset        | Task           | Description    | Reference      |
+================+================+================+================+
| C              | Classification | Cocitation     | `S             |
| ora-Cocitation |                | dataset.       | ource <https:/ |
|                |                |                | /proceedings.n |
|                |                |                | eurips.cc/pape |
|                |                |                | r_files/paper/ |
|                |                |                | 2019/file/1efa |
|                |                |                | 39bcaec6f39001 |
|                |                |                | 49160693694536 |
|                |                |                | -Paper.pdf>`__ |
+----------------+----------------+----------------+----------------+
| Cites          | Classification | Cocitation     | `S             |
| eer-Cocitation |                | dataset.       | ource <https:/ |
|                |                |                | /proceedings.n |
|                |                |                | eurips.cc/pape |
|                |                |                | r_files/paper/ |
|                |                |                | 2019/file/1efa |
|                |                |                | 39bcaec6f39001 |
|                |                |                | 49160693694536 |
|                |                |                | -Paper.pdf>`__ |
+----------------+----------------+----------------+----------------+
| Pub            | Classification | Cocitation     | `S             |
| Med-Cocitation |                | dataset.       | ource <https:/ |
|                |                |                | /proceedings.n |
|                |                |                | eurips.cc/pape |
|                |                |                | r_files/paper/ |
|                |                |                | 2019/file/1efa |
|                |                |                | 39bcaec6f39001 |
|                |                |                | 49160693694536 |
|                |                |                | -Paper.pdf>`__ |
+----------------+----------------+----------------+----------------+
| Cor            | Classification | Cocitation     | `S             |
| a-Coauthorship |                | dataset.       | ource <https:/ |
|                |                |                | /proceedings.n |
|                |                |                | eurips.cc/pape |
|                |                |                | r_files/paper/ |
|                |                |                | 2019/file/1efa |
|                |                |                | 39bcaec6f39001 |
|                |                |                | 49160693694536 |
|                |                |                | -Paper.pdf>`__ |
+----------------+----------------+----------------+----------------+
| DBL            | Classification | Cocitation     | `S             |
| P-Coauthorship |                | dataset.       | ource <https:/ |
|                |                |                | /proceedings.n |
|                |                |                | eurips.cc/pape |
|                |                |                | r_files/paper/ |
|                |                |                | 2019/file/1efa |
|                |                |                | 39bcaec6f39001 |
|                |                |                | 49160693694536 |
|                |                |                | -Paper.pdf>`__ |
+----------------+----------------+----------------+----------------+

References
~~~~~~~~~~~~~~~~

To learn more about ``TopoBench``, we invite you to read the paper:

::

   @article{telyatnikov2024topobench,
         title={TopoBench: A Framework for Benchmarking Topological Deep Learning},
         author={Lev Telyatnikov and Guillermo Bernardez and Marco Montagna and Pavlo Vasylenko and Ghada Zamzmi and Mustafa Hajij and Michael T Schaub and Nina Miolane and Simone Scardapane and Theodore Papamarkou},
         year={2024},
         eprint={2406.06642},
         archivePrefix={arXiv},
         primaryClass={cs.LG},
         url={https://arxiv.org/abs/2406.06642},
   }

If you find ``TopoBench`` useful, we would appreciate if you cite us!

Additional Details
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details>

Hierarchy of configuration files

::

   ├── configs                   <- Hydra configs
   │   ├── callbacks                <- Callbacks configs
   │   ├── dataset                  <- Dataset configs
   │   │   ├── graph                    <- Graph dataset configs
   │   │   ├── hypergraph               <- Hypergraph dataset configs
   │   │   └── simplicial               <- Simplicial dataset configs
   │   ├── debug                    <- Debugging configs
   │   ├── evaluator                <- Evaluator configs
   │   ├── experiment               <- Experiment configs
   │   ├── extras                   <- Extra utilities configs
   │   ├── hparams_search           <- Hyperparameter search configs
   │   ├── hydra                    <- Hydra configs
   │   ├── local                    <- Local configs
   │   ├── logger                   <- Logger configs
   │   ├── loss                     <- Loss function configs
   │   ├── model                    <- Model configs
   │   │   ├── cell                     <- Cell model configs
   │   │   ├── graph                    <- Graph model configs
   │   │   ├── hypergraph               <- Hypergraph model configs
   │   │   └── simplicial               <- Simplicial model configs
   │   ├── optimizer                <- Optimizer configs
   │   ├── paths                    <- Project paths configs
   │   ├── scheduler                <- Scheduler configs
   │   ├── trainer                  <- Trainer configs
   │   ├── transforms               <- Data transformation configs
   │   │   ├── data_manipulations       <- Data manipulation transforms
   │   │   ├── dataset_defaults         <- Default dataset transforms
   │   │   ├── feature_liftings         <- Feature lifting transforms
   │   │   └── liftings                 <- Lifting transforms
   │   │       ├── graph2cell               <- Graph to cell lifting transforms
   │   │       ├── graph2hypergraph         <- Graph to hypergraph lifting transforms
   │   │       ├── graph2simplicial         <- Graph to simplicial lifting transforms
   │   │       ├── graph2cell_default.yaml  <- Default graph to cell lifting config
   │   │       ├── graph2hypergraph_default.yaml <- Default graph to hypergraph lifting config
   │   │       ├── graph2simplicial_default.yaml <- Default graph to simplicial lifting config
   │   │       ├── no_lifting.yaml           <- No lifting config
   │   │       ├── custom_example.yaml       <- Custom example transform config
   │   │       └── no_transform.yaml         <- No transform config
   │   ├── wandb_sweep              <- Weights & Biases sweep configs
   │   │
   │   ├── __init__.py              <- Init file for configs module
   │   └── run.yaml               <- Main config for training

.. raw:: html

   </details>

.. raw:: html

   <details>

More information regarding Topological Deep Learning

`Topological Graph Signal
Compression <https://arxiv.org/pdf/2308.11068>`__

`Architectures of Topological Deep Learning: A Survey on Topological
Neural Networks <https://par.nsf.gov/servlets/purl/10477141>`__

`TopoX: a suite of Python packages for machine learning on topological
domains <https://arxiv.org/pdf/2402.02441>`__

.. raw:: html

   </details>

--------------

📢 Get in Touch!
~~~~~~~~~~~~~~~

| We are always open to collaborations and discussions on TDL research.
| Feel free to reach out via email if you want to collaborate, do your
  thesis with our team, or open a discussion for various opportunities.

| 📧 **Contact Email:** topological.intelligence@gmail.com
| ▶️ **YouTube Channel:** `Topological
  Intelligence <https://www.youtube.com/@TopologicalIntelligence>`__

.. |Lint| image:: https://github.com/geometric-intelligence/TopoBench/actions/workflows/lint.yml/badge.svg
   :target: https://github.com/geometric-intelligence/TopoBench/actions/workflows/lint.yml
.. |Test| image:: https://github.com/geometric-intelligence/TopoBench/actions/workflows/test.yml/badge.svg
   :target: https://github.com/geometric-intelligence/TopoBench/actions/workflows/test.yml
.. |Codecov| image:: https://codecov.io/gh/geometric-intelligence/TopoBench/branch/main/graph/badge.svg
   :target: https://app.codecov.io/gh/geometric-intelligence/TopoBench
.. |Docs| image:: https://img.shields.io/badge/docs-website-brightgreen
   :target: https://geometric-intelligence.github.io/topobench/index.html
.. |Python| image:: https://img.shields.io/badge/python-3.10+-blue?logo=python
   :target: https://www.python.org/
.. |license| image:: https://badgen.net/github/license/geometric-intelligence/TopoBench?color=green
   :target: https://github.com/geometric-intelligence/TopoBench/blob/main/LICENSE
.. |slack| image:: https://img.shields.io/badge/chat-on%20slack-purple?logo=slack
   :target: https://join.slack.com/t/geometric-intelligenceworkspace/shared_invite/zt-2k63sv99s-jbFMLtwzUCc8nt3sIRWjEw

.. toctree::
   :hidden:
   :maxdepth: 2

   tdl-challenge/index
   api/index
   contributing/index

Differentiable Neural Computers and family, for Pytorch
=======================================================

Includes: 1. Differentiable Neural Computers (DNC) 2. Sparse Access
Memory (SAM) 3. Sparse Differentiable Neural Computers (SDNC)

.. raw:: html

   <!-- START doctoc generated TOC please keep comment here to allow auto update -->

.. raw:: html

   <!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->

Table of Contents
-----------------

-  `Install <#install>`__

   -  `From source <#from-source>`__

-  `Architecure <#architecure>`__
-  `Usage <#usage>`__

   -  `DNC <#dnc>`__

      -  `Example usage <#example-usage>`__
      -  `Debugging <#debugging>`__

   -  `SDNC <#sdnc>`__

      -  `Example usage <#example-usage-1>`__
      -  `Debugging <#debugging-1>`__

   -  `SAM <#sam>`__

      -  `Example usage <#example-usage-2>`__
      -  `Debugging <#debugging-2>`__

-  `Tasks <#tasks>`__

   -  `Copy task (with curriculum and
      generalization) <#copy-task-with-curriculum-and-generalization>`__
   -  `Generalizing Addition task <#generalizing-addition-task>`__
   -  `Generalizing Argmax task <#generalizing-argmax-task>`__

-  `Code Structure <#code-structure>`__
-  `General noteworthy stuff <#general-noteworthy-stuff>`__

.. raw:: html

   <!-- END doctoc generated TOC please keep comment here to allow auto update -->

|Build Status| |PyPI version|

This is an implementation of `Differentiable Neural
Computers <http://people.idsia.ch/~rupesh/rnnsymposium2016/slides/graves.pdf>`__,
described in the paper `Hybrid computing using a neural network with
dynamic external memory, Graves et
al. <https://www.nature.com/articles/nature20101>`__ and Sparse DNCs
(SDNCs) and Sparse Access Memory (SAM) described in `Scaling
Memory-Augmented Neural Networks with Sparse Reads and
Writes <http://papers.nips.cc/paper/6298-scaling-memory-augmented-neural-networks-with-sparse-reads-and-writes.pdf>`__.

Install
-------

.. code:: bash

   pip install dnc

From source
~~~~~~~~~~~

::

   git clone https://github.com/ixaxaar/pytorch-dnc
   cd pytorch-dnc
   pip install -r ./requirements.txt
   pip install -e .

For using fully GPU based SDNCs or SAMs, install FAISS:

.. code:: bash

   conda install faiss-gpu -c pytorch

``pytest`` is required to run the test

Architecure
-----------

Usage
-----

DNC
~~~

**Constructor Parameters**:

Following are the constructor parameters:

Following are the constructor parameters:

+--------------+------+------------------------------------------------+
| Argument     | Def  | Description                                    |
|              | ault |                                                |
+==============+======+================================================+
| input_size   | ``No | Size of the input vectors                      |
|              | ne`` |                                                |
+--------------+------+------------------------------------------------+
| hidden_size  | ``No | Size of hidden units                           |
|              | ne`` |                                                |
+--------------+------+------------------------------------------------+
| rnn_type     | ``   | Type of recurrent cells used in the controller |
|              | 'lst |                                                |
|              | m'`` |                                                |
+--------------+------+------------------------------------------------+
| num_layers   | `    | Number of layers of recurrent units in the     |
|              | `1`` | controller                                     |
+--------------+------+------------------------------------------------+
| num_h        | `    | Number of hidden layers per layer of the       |
| idden_layers | `2`` | controller                                     |
+--------------+------+------------------------------------------------+
| bias         | ``Tr | Bias                                           |
|              | ue`` |                                                |
+--------------+------+------------------------------------------------+
| batch_first  | ``Tr | Whether data is fed batch first                |
|              | ue`` |                                                |
+--------------+------+------------------------------------------------+
| dropout      | `    | Dropout between layers in the controller       |
|              | `0`` |                                                |
+--------------+------+------------------------------------------------+
| b            | `    | If the controller is bidirectional (Not yet    |
| idirectional | `Fal | implemented                                    |
|              | se`` |                                                |
+--------------+------+------------------------------------------------+
| nr_cells     | `    | Number of memory cells                         |
|              | `5`` |                                                |
+--------------+------+------------------------------------------------+
| read_heads   | `    | Number of read heads                           |
|              | `2`` |                                                |
+--------------+------+------------------------------------------------+
| cell_size    | ``   | Size of each memory cell                       |
|              | 10`` |                                                |
+--------------+------+------------------------------------------------+
| nonlinearity | ``   | If using ‘rnn’ as ``rnn_type``, non-linearity  |
|              | 'tan | of the RNNs                                    |
|              | h'`` |                                                |
+--------------+------+------------------------------------------------+
| device       | ``   | ID of the GPU, -1 for CPU                      |
|              | -1`` |                                                |
+--------------+------+------------------------------------------------+
| indepen      | `    | Whether to use independent linear units to     |
| dent_linears | `Fal | derive interface vector                        |
|              | se`` |                                                |
+--------------+------+------------------------------------------------+
| share_memory | ``Tr | Whether to share memory between controller     |
|              | ue`` | layers                                         |
+--------------+------+------------------------------------------------+

Following are the forward pass parameters:

+-------------+------------+-------------------------------------------+
| Argument    | Default    | Description                               |
+=============+============+===========================================+
| input       | -          | The input vector ``(B*T*X)`` or           |
|             |            | ``(T*B*X)``                               |
+-------------+------------+-------------------------------------------+
| hidden      | ``(None,No | Hidden states                             |
|             | ne,None)`` | ``(controll                               |
|             |            | er hidden, memory hidden, read vectors)`` |
+-------------+------------+-------------------------------------------+
| reset       | ``False``  | Whether to reset memory                   |
| _experience |            |                                           |
+-------------+------------+-------------------------------------------+
| pass_thr    | ``True``   | Whether to pass through memory            |
| ough_memory |            |                                           |
+-------------+------------+-------------------------------------------+

Example usage
^^^^^^^^^^^^^

.. code:: python

   from dnc import DNC

   rnn = DNC(
     input_size=64,
     hidden_size=128,
     rnn_type='lstm',
     num_layers=4,
     nr_cells=100,
     cell_size=32,
     read_heads=4,
     batch_first=True,
     device=0
   )

   (controller_hidden, memory, read_vectors) = (None, None, None)

   output, (controller_hidden, memory, read_vectors) = \
     rnn(torch.randn(10, 4, 64), (controller_hidden, memory, read_vectors), reset_experience=True)

Debugging
^^^^^^^^^

The ``debug`` option causes the network to return its memory hidden
vectors (numpy ``ndarray``\ s) for the first batch each forward step.
These vectors can be analyzed or visualized, using visdom for example.

.. code:: python

   from dnc import DNC

   rnn = DNC(
     input_size=64,
     hidden_size=128,
     rnn_type='lstm',
     num_layers=4,
     nr_cells=100,
     cell_size=32,
     read_heads=4,
     batch_first=True,
     device=0,
     debug=True
   )

   (controller_hidden, memory, read_vectors) = (None, None, None)

   output, (controller_hidden, memory, read_vectors), debug_memory = \
     rnn(torch.randn(10, 4, 64), (controller_hidden, memory, read_vectors), reset_experience=True)

Memory vectors returned by forward pass (``np.ndarray``):

+----------------------+---------------------+----------------------+
| Key                  | Y axis (dimensions) | X axis (dimensions)  |
+======================+=====================+======================+
| ``debu               | layer \* time       | nr_cells \*          |
| g_memory['memory']`` |                     | cell_size            |
+----------------------+---------------------+----------------------+
| ``debug_mem          | layer \* time       | nr_cells \* nr_cells |
| ory['link_matrix']`` |                     |                      |
+----------------------+---------------------+----------------------+
| ``debug_me           | layer \* time       | nr_cells             |
| mory['precedence']`` |                     |                      |
+----------------------+---------------------+----------------------+
| ``debug_memo         | layer \* time       | read_heads \*        |
| ry['read_weights']`` |                     | nr_cells             |
+----------------------+---------------------+----------------------+
| ``debug_memor        | layer \* time       | nr_cells             |
| y['write_weights']`` |                     |                      |
+----------------------+---------------------+----------------------+
| ``debug_memo         | layer \* time       | nr_cells             |
| ry['usage_vector']`` |                     |                      |
+----------------------+---------------------+----------------------+

SDNC
~~~~

**Constructor Parameters**:

Following are the constructor parameters:

+--------------+------+------------------------------------------------+
| Argument     | Def  | Description                                    |
|              | ault |                                                |
+==============+======+================================================+
| input_size   | ``No | Size of the input vectors                      |
|              | ne`` |                                                |
+--------------+------+------------------------------------------------+
| hidden_size  | ``No | Size of hidden units                           |
|              | ne`` |                                                |
+--------------+------+------------------------------------------------+
| rnn_type     | ``   | Type of recurrent cells used in the controller |
|              | 'lst |                                                |
|              | m'`` |                                                |
+--------------+------+------------------------------------------------+
| num_layers   | `    | Number of layers of recurrent units in the     |
|              | `1`` | controller                                     |
+--------------+------+------------------------------------------------+
| num_h        | `    | Number of hidden layers per layer of the       |
| idden_layers | `2`` | controller                                     |
+--------------+------+------------------------------------------------+
| bias         | ``Tr | Bias                                           |
|              | ue`` |                                                |
+--------------+------+------------------------------------------------+
| batch_first  | ``Tr | Whether data is fed batch first                |
|              | ue`` |                                                |
+--------------+------+------------------------------------------------+
| dropout      | `    | Dropout between layers in the controller       |
|              | `0`` |                                                |
+--------------+------+------------------------------------------------+
| b            | `    | If the controller is bidirectional (Not yet    |
| idirectional | `Fal | implemented                                    |
|              | se`` |                                                |
+--------------+------+------------------------------------------------+
| nr_cells     | ``50 | Number of memory cells                         |
|              | 00`` |                                                |
+--------------+------+------------------------------------------------+
| read_heads   | `    | Number of read heads                           |
|              | `4`` |                                                |
+--------------+------+------------------------------------------------+
| sparse_reads | `    | Number of sparse memory reads per read head    |
|              | `4`` |                                                |
+--------------+------+------------------------------------------------+
| te           | `    | Number of temporal reads                       |
| mporal_reads | `4`` |                                                |
+--------------+------+------------------------------------------------+
| cell_size    | ``   | Size of each memory cell                       |
|              | 10`` |                                                |
+--------------+------+------------------------------------------------+
| nonlinearity | ``   | If using ‘rnn’ as ``rnn_type``, non-linearity  |
|              | 'tan | of the RNNs                                    |
|              | h'`` |                                                |
+--------------+------+------------------------------------------------+
| device       | ``   | ID of the GPU, -1 for CPU                      |
|              | -1`` |                                                |
+--------------+------+------------------------------------------------+
| indepen      | `    | Whether to use independent linear units to     |
| dent_linears | `Fal | derive interface vector                        |
|              | se`` |                                                |
+--------------+------+------------------------------------------------+
| share_memory | ``Tr | Whether to share memory between controller     |
|              | ue`` | layers                                         |
+--------------+------+------------------------------------------------+

Following are the forward pass parameters:

+-------------+------------+-------------------------------------------+
| Argument    | Default    | Description                               |
+=============+============+===========================================+
| input       | -          | The input vector ``(B*T*X)`` or           |
|             |            | ``(T*B*X)``                               |
+-------------+------------+-------------------------------------------+
| hidden      | ``(None,No | Hidden states                             |
|             | ne,None)`` | ``(controll                               |
|             |            | er hidden, memory hidden, read vectors)`` |
+-------------+------------+-------------------------------------------+
| reset       | ``False``  | Whether to reset memory                   |
| _experience |            |                                           |
+-------------+------------+-------------------------------------------+
| pass_thr    | ``True``   | Whether to pass through memory            |
| ough_memory |            |                                           |
+-------------+------------+-------------------------------------------+

.. _example-usage-1:

Example usage
^^^^^^^^^^^^^

.. code:: python

   from dnc import SDNC

   rnn = SDNC(
     input_size=64,
     hidden_size=128,
     rnn_type='lstm',
     num_layers=4,
     nr_cells=100,
     cell_size=32,
     read_heads=4,
     sparse_reads=4,
     batch_first=True,
     device=0
   )

   (controller_hidden, memory, read_vectors) = (None, None, None)

   output, (controller_hidden, memory, read_vectors) = \
     rnn(torch.randn(10, 4, 64), (controller_hidden, memory, read_vectors), reset_experience=True)

.. _debugging-1:

Debugging
^^^^^^^^^

The ``debug`` option causes the network to return its memory hidden
vectors (numpy ``ndarray``\ s) for the first batch each forward step.
These vectors can be analyzed or visualized, using visdom for example.

.. code:: python

   from dnc import SDNC

   rnn = SDNC(
     input_size=64,
     hidden_size=128,
     rnn_type='lstm',
     num_layers=4,
     nr_cells=100,
     cell_size=32,
     read_heads=4,
     batch_first=True,
     sparse_reads=4,
     temporal_reads=4,
     device=0,
     debug=True
   )

   (controller_hidden, memory, read_vectors) = (None, None, None)

   output, (controller_hidden, memory, read_vectors), debug_memory = \
     rnn(torch.randn(10, 4, 64), (controller_hidden, memory, read_vectors), reset_experience=True)

Memory vectors returned by forward pass (``np.ndarray``):

+-----------------------+-----------------------+-----------------------+
| Key                   | Y axis (dimensions)   | X axis (dimensions)   |
+=======================+=======================+=======================+
| ``deb                 | layer \* time         | nr_cells \* cell_size |
| ug_memory['memory']`` |                       |                       |
+-----------------------+-----------------------+-----------------------+
| ``debug_memor         | layer \* time         | sparse_reads+         |
| y['visible_memory']`` |                       | 2\ *temporal_reads+1* |
|                       |                       | nr_cells              |
+-----------------------+-----------------------+-----------------------+
| ``debug_memor         | layer \* time         | sparse_rea            |
| y['read_positions']`` |                       | ds+2*temporal_reads+1 |
+-----------------------+-----------------------+-----------------------+
| ``debug_me            | layer \* time         | sparse_reads+         |
| mory['link_matrix']`` |                       | 2\ *temporal_reads+1* |
|                       |                       | sparse_rea            |
|                       |                       | ds+2*temporal_reads+1 |
+-----------------------+-----------------------+-----------------------+
| ``debug_memory        | layer \* time         | sparse_reads+         |
| ['rev_link_matrix']`` |                       | 2\ *temporal_reads+1* |
|                       |                       | sparse_rea            |
|                       |                       | ds+2*temporal_reads+1 |
+-----------------------+-----------------------+-----------------------+
| ``debug_m             | layer \* time         | nr_cells              |
| emory['precedence']`` |                       |                       |
+-----------------------+-----------------------+-----------------------+
| ``debug_mem           | layer \* time         | read_heads \*         |
| ory['read_weights']`` |                       | nr_cells              |
+-----------------------+-----------------------+-----------------------+
| ``debug_memo          | layer \* time         | nr_cells              |
| ry['write_weights']`` |                       |                       |
+-----------------------+-----------------------+-----------------------+
| ``de                  | layer \* time         | nr_cells              |
| bug_memory['usage']`` |                       |                       |
+-----------------------+-----------------------+-----------------------+

SAM
~~~

**Constructor Parameters**:

Following are the constructor parameters:

+--------------+------+------------------------------------------------+
| Argument     | Def  | Description                                    |
|              | ault |                                                |
+==============+======+================================================+
| input_size   | ``No | Size of the input vectors                      |
|              | ne`` |                                                |
+--------------+------+------------------------------------------------+
| hidden_size  | ``No | Size of hidden units                           |
|              | ne`` |                                                |
+--------------+------+------------------------------------------------+
| rnn_type     | ``   | Type of recurrent cells used in the controller |
|              | 'lst |                                                |
|              | m'`` |                                                |
+--------------+------+------------------------------------------------+
| num_layers   | `    | Number of layers of recurrent units in the     |
|              | `1`` | controller                                     |
+--------------+------+------------------------------------------------+
| num_h        | `    | Number of hidden layers per layer of the       |
| idden_layers | `2`` | controller                                     |
+--------------+------+------------------------------------------------+
| bias         | ``Tr | Bias                                           |
|              | ue`` |                                                |
+--------------+------+------------------------------------------------+
| batch_first  | ``Tr | Whether data is fed batch first                |
|              | ue`` |                                                |
+--------------+------+------------------------------------------------+
| dropout      | `    | Dropout between layers in the controller       |
|              | `0`` |                                                |
+--------------+------+------------------------------------------------+
| b            | `    | If the controller is bidirectional (Not yet    |
| idirectional | `Fal | implemented                                    |
|              | se`` |                                                |
+--------------+------+------------------------------------------------+
| nr_cells     | ``50 | Number of memory cells                         |
|              | 00`` |                                                |
+--------------+------+------------------------------------------------+
| read_heads   | `    | Number of read heads                           |
|              | `4`` |                                                |
+--------------+------+------------------------------------------------+
| sparse_reads | `    | Number of sparse memory reads per read head    |
|              | `4`` |                                                |
+--------------+------+------------------------------------------------+
| cell_size    | ``   | Size of each memory cell                       |
|              | 10`` |                                                |
+--------------+------+------------------------------------------------+
| nonlinearity | ``   | If using ‘rnn’ as ``rnn_type``, non-linearity  |
|              | 'tan | of the RNNs                                    |
|              | h'`` |                                                |
+--------------+------+------------------------------------------------+
| device       | ``   | ID of the GPU, -1 for CPU                      |
|              | -1`` |                                                |
+--------------+------+------------------------------------------------+
| indepen      | `    | Whether to use independent linear units to     |
| dent_linears | `Fal | derive interface vector                        |
|              | se`` |                                                |
+--------------+------+------------------------------------------------+
| share_memory | ``Tr | Whether to share memory between controller     |
|              | ue`` | layers                                         |
+--------------+------+------------------------------------------------+

Following are the forward pass parameters:

+-------------+------------+-------------------------------------------+
| Argument    | Default    | Description                               |
+=============+============+===========================================+
| input       | -          | The input vector ``(B*T*X)`` or           |
|             |            | ``(T*B*X)``                               |
+-------------+------------+-------------------------------------------+
| hidden      | ``(None,No | Hidden states                             |
|             | ne,None)`` | ``(controll                               |
|             |            | er hidden, memory hidden, read vectors)`` |
+-------------+------------+-------------------------------------------+
| reset       | ``False``  | Whether to reset memory                   |
| _experience |            |                                           |
+-------------+------------+-------------------------------------------+
| pass_thr    | ``True``   | Whether to pass through memory            |
| ough_memory |            |                                           |
+-------------+------------+-------------------------------------------+

.. _example-usage-2:

Example usage
^^^^^^^^^^^^^

.. code:: python

   from dnc import SAM

   rnn = SAM(
     input_size=64,
     hidden_size=128,
     rnn_type='lstm',
     num_layers=4,
     nr_cells=100,
     cell_size=32,
     read_heads=4,
     sparse_reads=4,
     batch_first=True,
     device=0
   )

   (controller_hidden, memory, read_vectors) = (None, None, None)

   output, (controller_hidden, memory, read_vectors) = \
     rnn(torch.randn(10, 4, 64), (controller_hidden, memory, read_vectors), reset_experience=True)

.. _debugging-2:

Debugging
^^^^^^^^^

The ``debug`` option causes the network to return its memory hidden
vectors (numpy ``ndarray``\ s) for the first batch each forward step.
These vectors can be analyzed or visualized, using visdom for example.

.. code:: python

   from dnc import SAM

   rnn = SAM(
     input_size=64,
     hidden_size=128,
     rnn_type='lstm',
     num_layers=4,
     nr_cells=100,
     cell_size=32,
     read_heads=4,
     batch_first=True,
     sparse_reads=4,
     device=0,
     debug=True
   )

   (controller_hidden, memory, read_vectors) = (None, None, None)

   output, (controller_hidden, memory, read_vectors), debug_memory = \
     rnn(torch.randn(10, 4, 64), (controller_hidden, memory, read_vectors), reset_experience=True)

Memory vectors returned by forward pass (``np.ndarray``):

+-----------------------+-----------------------+-----------------------+
| Key                   | Y axis (dimensions)   | X axis (dimensions)   |
+=======================+=======================+=======================+
| ``deb                 | layer \* time         | nr_cells \* cell_size |
| ug_memory['memory']`` |                       |                       |
+-----------------------+-----------------------+-----------------------+
| ``debug_memor         | layer \* time         | sparse_reads+         |
| y['visible_memory']`` |                       | 2\ *temporal_reads+1* |
|                       |                       | nr_cells              |
+-----------------------+-----------------------+-----------------------+
| ``debug_memor         | layer \* time         | sparse_rea            |
| y['read_positions']`` |                       | ds+2*temporal_reads+1 |
+-----------------------+-----------------------+-----------------------+
| ``debug_mem           | layer \* time         | read_heads \*         |
| ory['read_weights']`` |                       | nr_cells              |
+-----------------------+-----------------------+-----------------------+
| ``debug_memo          | layer \* time         | nr_cells              |
| ry['write_weights']`` |                       |                       |
+-----------------------+-----------------------+-----------------------+
| ``de                  | layer \* time         | nr_cells              |
| bug_memory['usage']`` |                       |                       |
+-----------------------+-----------------------+-----------------------+

Tasks
-----

Copy task (with curriculum and generalization)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The copy task, as descibed in the original paper, is included in the
repo.

From the project root:

.. code:: bash

   python ./tasks/copy_task.py -cuda 0 -optim rmsprop -batch_size 32 -mem_slot 64 # (like original implementation)

   python ./tasks/copy_task.py -cuda 0 -lr 0.001 -rnn_type lstm -nlayer 1 -nhlayer 2 -dropout 0 -mem_slot 32 -batch_size 1000 -optim adam -sequence_max_length 8 # (faster convergence)

   For SDNCs:
   python ./tasks/copy_task.py -cuda 0 -lr 0.001 -rnn_type lstm -memory_type sdnc -nlayer 1 -nhlayer 2 -dropout 0 -mem_slot 100 -mem_size 10  -read_heads 1 -sparse_reads 10 -batch_size 20 -optim adam -sequence_max_length 10

   and for curriculum learning for SDNCs:
   python ./tasks/copy_task.py -cuda 0 -lr 0.001 -rnn_type lstm -memory_type sdnc -nlayer 1 -nhlayer 2 -dropout 0 -mem_slot 100 -mem_size 10  -read_heads 1 -sparse_reads 4 -temporal_reads 4 -batch_size 20 -optim adam -sequence_max_length 4 -curriculum_increment 2 -curriculum_freq 10000

For the full set of options, see:

::

   python ./tasks/copy_task.py --help

The copy task can be used to debug memory using
`Visdom <https://github.com/facebookresearch/visdom>`__.

Additional step required:

.. code:: bash

   pip install visdom
   python -m visdom.server

Open http://localhost:8097/ on your browser, and execute the copy task:

.. code:: bash

   python ./tasks/copy_task.py -cuda 0

The visdom dashboard shows memory as a heatmap for batch 0 every
``-summarize_freq`` iteration:

.. figure:: ./docs/dnc-mem-debug.png
   :alt: Visdom dashboard

   Visdom dashboard

Generalizing Addition task
~~~~~~~~~~~~~~~~~~~~~~~~~~

The adding task is as described in `this github pull
request <https://github.com/Mostafa-Samir/DNC-tensorflow/pull/4#issue-199369192>`__.
This task - creates one-hot vectors of size ``input_size``, each
representing a number - feeds a sentence of them to a network - the
output of which is added to get the sum of the decoded outputs

The task first trains the network for sentences of size ~100, and then
tests if the network genetalizes for lengths ~1000.

.. code:: bash

   python ./tasks/adding_task.py -cuda 0 -lr 0.0001 -rnn_type lstm -memory_type sam -nlayer 1 -nhlayer 1 -nhid 100 -dropout 0 -mem_slot 1000 -mem_size 32 -read_heads 1 -sparse_reads 4 -batch_size 20 -optim rmsprop -input_size 3 -sequence_max_length 100

Generalizing Argmax task
~~~~~~~~~~~~~~~~~~~~~~~~

The second adding task is similar to the first one, except that the
network’s output at the last time step is expected to be the argmax of
the input.

.. code:: bash

   python ./tasks/argmax_task.py -cuda 0 -lr 0.0001 -rnn_type lstm -memory_type dnc -nlayer 1 -nhlayer 1 -nhid 100 -dropout 0 -mem_slot 100 -mem_size 10 -read_heads 2 -batch_size 1 -optim rmsprop -sequence_max_length 15 -input_size 10 -iterations 10000

Code Structure
--------------

1. DNCs:

-  `dnc/dnc.py <dnc/dnc.py>`__ - Controller code.
-  `dnc/memory.py <dnc/memory.py>`__ - Memory module.

2. SDNCs:

-  `dnc/sdnc.py <dnc/sdnc.py>`__ - Controller code, inherits
   `dnc.py <dnc/dnc.py>`__.
-  `dnc/sparse_temporal_memory.py <dnc/sparse_temporal_memory.py>`__ -
   Memory module.
-  `dnc/flann_index.py <dnc/flann_index.py>`__ - Memory index using kNN.

3. SAMs:

-  `dnc/sam.py <dnc/sam.py>`__ - Controller code, inherits
   `dnc.py <dnc/dnc.py>`__.
-  `dnc/sparse_memory.py <dnc/sparse_memory.py>`__ - Memory module.
-  `dnc/flann_index.py <dnc/flann_index.py>`__ - Memory index using kNN.

4. Tests:

-  All tests are in `./tests <./tests>`__ folder.

General noteworthy stuff
------------------------

1. SDNCs use the `FLANN approximate nearest neigbhour
   library <https://github.com/mariusmuja/flann>`__, with its python
   binding `pyflann3 <https://github.com/primetang/pyflann>`__ and
   `FAISS <https://github.com/facebookresearch/faiss>`__.

FLANN can be installed either from pip (automatically as a dependency),
or from source (e.g. for multithreading via OpenMP):

.. code:: bash

   # install openmp first: e.g. `sudo pacman -S openmp` for Arch.
   git clone git://github.com/mariusmuja/flann.git
   cd flann
   mkdir build
   cd build
   cmake ..
   make -j 4
   sudo make install

FAISS can be installed using:

.. code:: bash

   conda install faiss-gpu -c pytorch

FAISS is much faster, has a GPU implementation and is interoperable with
pytorch tensors. We try to use FAISS by default, in absence of which we
fall back to FLANN.

2. ``nan``\ s in the gradients are common, try with different batch
   sizes

Repos referred to for creation of this repo:

-  `deepmind/dnc <https://github.com/deepmind/dnc>`__
-  `ypxie/pytorch-NeuCom <https://github.com/ypxie/pytorch-NeuCom>`__
-  `jingweiz/pytorch-dnc <https://github.com/jingweiz/pytorch-dnc>`__

.. |Build Status| image:: https://travis-ci.org/ixaxaar/pytorch-dnc.svg?branch=master
   :target: https://travis-ci.org/ixaxaar/pytorch-dnc
.. |PyPI version| image:: https://badge.fury.io/py/dnc.svg
   :target: https://badge.fury.io/py/dnc

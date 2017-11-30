# Differentiable Neural Computers and Sparse Differentiable Neural Computers, for Pytorch

[![Build Status](https://travis-ci.org/ixaxaar/pytorch-dnc.svg?branch=master)](https://travis-ci.org/ixaxaar/pytorch-dnc) [![PyPI version](https://badge.fury.io/py/dnc.svg)](https://badge.fury.io/py/dnc)

This is an implementation of [Differentiable Neural Computers](http://people.idsia.ch/~rupesh/rnnsymposium2016/slides/graves.pdf), described in the paper [Hybrid computing using a neural network with dynamic external memory, Graves et al.](https://www.nature.com/articles/nature20101)
and the Sparse version of the DNC (the SDNC) described in [Scaling Memory-Augmented Neural Networks with Sparse Reads and Writes](http://papers.nips.cc/paper/6298-scaling-memory-augmented-neural-networks-with-sparse-reads-and-writes.pdf).

## Install

```bash
pip install dnc
```

For using sparse DNCs, additional libraries are required:

### FAISS

SDNCs require an additional library: [facebookresearch/faiss](https://github.com/facebookresearch/faiss).
A compiled version of the library with intel SSE + CUDA 8 support ships with this library.
If that does not work, one might need to manually compile faiss, as detailed below:

#### Installing FAISS

Needs `libopenblas.so` in `/usr/lib/`.

This has been tested on Arch Linux. Other distributions might have different libopenblas path or cuda root dir or numpy include files dir.

```bash
git clone https://github.com/facebookresearch/faiss.git
cd faiss
cp ./example_makefiles/makefile.inc.Linux ./makefile.inc
# change libopenblas path
sed -i "s/lib64\/libopenblas\.so\.0/lib\/libopenblas\.so/g" ./makefile.inc
# add option for nvcc to work properly with g++ > 5
sed -i "s/std c++11 \-lineinfo/std c++11 \-lineinfo \-Xcompiler \-D__CORRECT_ISO_CPP11_MATH_H_PROTO/g" ./makefile.inc
# change CUDA ROOT
sed -i "s/CUDAROOT=\/usr\/local\/cuda-8.0\//CUDAROOT=\/opt\/cuda\//g" ./makefile.inc
# change numpy include files (for v3.6)
sed -i "s/PYTHONCFLAGS=\-I\/usr\/include\/python2.7\/ \-I\/usr\/lib64\/python2.7\/site\-packages\/numpy\/core\/include\//PYTHONCFLAGS=\-I\/usr\/include\/python3.6m\/ \-I\/usr\/lib\/python3.6\/site\-packages\/numpy\/core\/include/g"

# build
make
cd gpu
make
cd ..
make py
cd gpu
make py
cd ..

mkdir /tmp/faiss
find -name "*.so" -exec cp {} /tmp/faiss \;
find -name "*.a" -exec cp {} /tmp/faiss \;
find -name "*.py" -exec cp {} /tmp/faiss \;
mv /tmp/faiss .
cd faiss

# convert to python3
2to3 -w ./*.py
rm -rf *.bak

# Fix relative imports
for i in *.py; do
  filename=`echo $i | cut -d "." -f 1`
  echo $filename
  find -name "*.py" -exec sed -i "s/import $filename/import \.$filename/g" {} \;
  find -name "*.py" -exec sed -i "s/from $filename import/from \.$filename import/g" {} \;
done

cd ..

git clone https://github.com/ixaxaar/pytorch-dnc
mv faiss pytorch-dnc
cd pytorch-dnc
sudo pip install -e .
```




## Architecure

<img src="./docs/dnc.png" height="600" />

## Usage

**Parameters**:

Following are the constructor parameters:

| Argument | Default | Description |
| --- | --- | --- |
| input_size | `None` | Size of the input vectors |
| hidden_size | `None` | Size of hidden units |
| rnn_type | `'lstm'` | Type of recurrent cells used in the controller |
| num_layers | `1` | Number of layers of recurrent units in the controller |
| num_hidden_layers | `2` | Number of hidden layers per layer of the controller |
| bias | `True` | Bias |
| batch_first | `True` | Whether data is fed batch first |
| dropout | `0` | Dropout between layers in the controller |
| bidirectional | `False` | If the controller is bidirectional (Not yet implemented |
| nr_cells | `5` | Number of memory cells |
| read_heads | `2` | Number of read heads |
| cell_size | `10` | Size of each memory cell |
| nonlinearity | `'tanh'` | If using 'rnn' as `rnn_type`, non-linearity of the RNNs |
| gpu_id | `-1` | ID of the GPU, -1 for CPU |
| independent_linears | `False` | Whether to use independent linear units to derive interface vector |
| share_memory | `True` | Whether to share memory between controller layers |

Following are the forward pass parameters:

| Argument | Default | Description |
| --- | --- | --- |
| input | - | The input vector `(B*T*X)` or `(T*B*X)` |
| hidden | `(None,None,None)` | Hidden states `(controller hidden, memory hidden, read vectors)` |
| reset_experience | `False` | Whether to reset memory (This is a parameter for the forward pass |
| pass_through_memory | `True` | Whether to pass through memory (This is a parameter for the forward pass |


### Example usage:

```python
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
  gpu_id=0
)

(controller_hidden, memory, read_vectors) = (None, None, None)

output, (controller_hidden, memory, read_vectors) = \
  rnn(torch.randn(10, 4, 64), (controller_hidden, memory, read_vectors, reset_experience=True))
```

### Debugging:

The `debug` option causes the network to return its memory hidden vectors (numpy `ndarray`s) for the first batch each forward step.
These vectors can be analyzed or visualized, using visdom for example.

```python
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
  gpu_id=0,
  debug=True
)

(controller_hidden, memory, read_vectors) = (None, None, None)

output, (controller_hidden, memory, read_vectors), debug_memory = \
  rnn(torch.randn(10, 4, 64), (controller_hidden, memory, read_vectors, reset_experience=True))
```

Memory vectors returned by forward pass (`np.ndarray`):

| Key | Y axis (dimensions) | X axis (dimensions) |
| --- | --- | --- |
| `debug_memory['memory']` | layer * time | nr_cells * cell_size
| `debug_memory['link_matrix']` | layer * time | nr_cells * nr_cells
| `debug_memory['precedence']` | layer * time | nr_cells
| `debug_memory['read_weights']` | layer * time | read_heads * nr_cells
| `debug_memory['write_weights']` | layer * time | nr_cells
| `debug_memory['usage_vector']` | layer * time | nr_cells

## Example copy task

The copy task, as descibed in the original paper, is included in the repo.

From the project root:
```bash
python ./tasks/copy_task.py -cuda 0 -optim rmsprop -batch_size 32 -mem_slot 64 # (like original implementation)

python3 ./tasks/copy_task.py -cuda 0 -lr 0.001 -rnn_type lstm -nlayer 1 -nhlayer 2 -dropout 0 -mem_slot 32 -batch_size 1000 -optim adam -sequence_max_length 8 # (faster convergence)
```

For the full set of options, see:
```
python ./tasks/copy_task.py --help
```

The copy task can be used to debug memory using [Visdom](https://github.com/facebookresearch/visdom).

Additional step required:

```bash
pip install visdom
python -m visdom.server
```

Open http://localhost:8097/ on your browser, and execute the copy task:

```bash
python ./tasks/copy_task.py -cuda 0
```

The visdom dashboard shows memory as a heatmap for batch 0 every `-summarize_freq` iteration:

![Visdom dashboard](./docs/dnc-mem-debug.png)


## General noteworthy stuff

1. DNCs converge faster with Adam and RMSProp learning rules, SGD generally converges extremely slowly.
The copy task, for example, takes 25k iterations on SGD with lr 1 compared to 3.5k for adam with lr 0.01.
2. `nan`s in the gradients are common, try with different batch sizes

Repos referred to for creation of this repo:

- [deepmind/dnc](https://github.com/deepmind/dnc)
- [ypxie/pytorch-NeuCom](https://github.com/ypxie/pytorch-NeuCom)
- [jingweiz/pytorch-dnc](https://github.com/jingweiz/pytorch-dnc)


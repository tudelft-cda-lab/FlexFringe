# Active learning library (BETA)

## Information

This is the active learning library of flexfringe. It has been designed to accommodate development of novel algorithms and enables the integration of active and passive learning. Examples of these are the Ldot and the PAUl algorithm.
Additionally, its implementation in C++ and the efficient use of data structures enables fast executions. However, there are still optimizations to be done. 

**Remark to BETA version:** Since our last overhaul many changes have been done to tried and tested algorithms, such as the weighted L# algorithm and the Ldot algorithm. We currently do not have the recourses for extensive testing, therefore if you find 

- Bugs or
- Points of improvement
- Features you would like to have

please do not hesitate to contact us via mail: s.e.verwer@tudelft.nl

## Structure

High level structure of this sub-project inspired by AALpy library: https://github.com/DES-Lab/AALpy

About the folder structure:
  1. Algorithms: Here we simply do implement the algorithms as they are.
  2. Teacher: This is the teacher. We chose to separate the teacher from the equivalence oracle to have maximum flexibility in our implementation.
  3. Oracle: A general oracle, providing both the membership-queries as well as the equivalence-queries. It supports multiple search- and processing-strategies for counterexamples. 
  4. Memory: Some sort of cache or other utility structures that store history and the databases to query from. Contains the observation tables as well.
  5. System under learning: This is an abstraction of the system. It potentially needs to be implemented for your use case. We provide abstract base class.
  7. Databases: Here we provide the database, mainly in conjunction with streaming, not for standalone active learning algorithms.
  8. Util (active_learning_util): Here we do some typedefs and some shared variables that a reasonable number of algorithms will use.

Notes on code (for those who want to dive deeper):
- We use the terms SUL (system under learning) and SUT (system under test). They mean the same for us.

## Learning (Distilling) Neural Networks

When learning neural networks, python scripts have to be used. We provide a template in the subdirectory system_under_learning/neural_network_suls/python/nn_connector_template.py,
for reference please consult it's doc-strings and the already implemented scripts. The python script is a wrapper around the network. **Important**: Since
flexfringe needs to know what kind of strings it can ask the network, we need to load an alphabet. Make sure that you understand how we infer the alphabet. For example, for some problems we require the alphabet to reside in the same directory as the model that we want to load.

To run Neural nets with flexfringe (e.g. query them using active learning) the --ini-file parameter is the casual one, but we expect as the input-file the
file to the python script that you want to execute. The other parameters are as follows:

- aptafile: This is the relative path from the executable to the network. It will be provided to the python script upon loading the network.
- input-file: The last argument of flexfringe by convention. This is the relative path to the python-script.

**Important**: We highly suggest you test and debug your python scripts first if you intend to write your own custom ones, because they are easier to spot and debug directly on the script.

**Pitfall in PyTorch**: When inferring in the model, both model.eval() and with torch.no_grad() should be used, else correct output of the model
seems not guaranteed anymore.

## Debugging

- We first recommend running the script as a standalone in Python, where you can perform the debugging in Python only using e.g. pdb. 
- If you think the problem is on the .cpp side, such as memory leaks, you can use Valgrind, see the next point.
- To run valgrind on the Python interface, download the valgrind-python.supp file from the [cpython-github](https://github.com/python/cpython/blob/main/Misc/valgrind-python.supp), and the run it using the flag

    --suppressions=\[some path\]/valgrind-python.supp
High level structure of this sub-project inspired by AALpy library: https://github.com/DES-Lab/AALpy

About the folder structure: 
  1. Algorithms: Here we simply do implement the algorithms as they are. 
  2. Teacher: This is the teacher. We chose to separate the teacher from the equivalence oracle to have maximum flexibility in our implementation.
  3. Equivalence oracle: Self explaining. We made it separate so we can experiment with different strategies. 
  4. Memory: Some sort of cache or other utility structures that store history and the databases to query from. Contains the observation tables as well. 
  5. System under learning: This is an abstraction of the system. It potentially needs to be implemented for your use case. We provide abstract base class. 
  6. Counterexample strategies: This is what you think it is.
  7. Databases: Here we provide the database, mainly in conjunction with streaming, not for standalone active learning algorithms.
  8. Util: Here we do some typedefs and some shared variables that a reasonable number of algorithms will use.

We use Umbrello to open the class diagram. **Note:** The class diagram is not maintained at the moment.

Notes on code (for those who want to dive deeper): 
- We use the terms SUL (system under learning) and SUT (system under test). They mean the same for us.

# Learning (Distilling) Neural Networks

When learning neural networks, python scripts have to be used. We provide a template in the subdirectory system_under_learning/python/nn_connector_template.py,
for reference please consult it's doc-strings and the already implemented scripts. The python script is a wrapper around the network. **Important**: Since
flexfringe needs to know what kind of strings it can ask the network, we need to load an alphabet. At the moment we require the alphabet to reside in the 
same directory as the model that we want to load.

To run Neural nets with flexfringe (e.g. query them using active learning) the --ini-file parameter is the casual one, but we expect as the input-file the 
file to the python script that you want to execute. The other parameters are as follows:

- aptafile: This is the relative path from the executable to the network. It will be provided to the python script upon loading the network.
- input-file: The last argument of flexfringe by convention. This is the relative path to the python-script.
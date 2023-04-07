High level structure of this sub-project inspired by AALpy library: https://github.com/DES-Lab/AALpy

About the folder structure: 
  1. Algorithms: Here we simply do implement the algorithms as they are. 
  2. Teacher: This is the teacher. We chose to separate the teacher from the equivalence oracle to have maximum flexibility in our implementation.
  3. Equivalence oracle: Self explaining. We made it separate so we can experiment with different strategies. 
  4. Memory: Some sort of cache or other utility structures that store history and the databases to query from. Contains the observation tables as well. 
  5. System under learning: This is an abstraction of the system. It potentially needs to be implemented for your use case. We provide abstract base class. 
  6. Counterexample strategies: This is what you think it is.
  7. Util: Here we do some typedefs and some shared variables that a reasonable number of algorithms will use.

We use Umbrello to open the class diagram.

Notes on code (for those who want to dive deeper): 
- We use the terms SUL (system under learning) and SUT (system under test). They mean the same for us.
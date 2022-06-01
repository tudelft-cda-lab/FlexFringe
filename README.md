# README #

flexfringe (formerly DFASAT), a flexible state-merging framework written in C++.

## What this repositor contains ##

This repository contains the latest release version of flexfringe.

## How to get set up ##

flexfringe compiles without external dependencies. It currently supports build chains using make and cmake.

For expert users: In case you want to use the reduction to SAT and automatically invoke the SAT solver, you need to provide the path to the solver binary. flexfringe has been tested with lingeling (which you can get from http://fmv.jku.at/lingeling/ and run its build.sh).
**PLEASE NOTE:** SAT solving only works for learning plain DFAs. The current implementation is not verified to be correct. Use an older commit if you rely on SAT-solving.

You can build and compile the flexfringe project by running

`$ make clean all`

or alternatively, using CMake

`$ mkdir build && cd build && cmake ..`
`$ make`

in the main directory to build the executable named *flexfringe*. There is also a CMakelists.txt for building with cmake. We tested the toolchains on Linux (Ubuntu 16+), MacOS (10.14), and Windows 10. For the latter, be built using CMake shipped with CLion.

## How to run it ##

Run ./flexfringe --help to get help.

We provide several .ini files as a shortcut to storing commonly used settings.

Example:

`$ ./flexfringe --ini ini/batch-overlap.ini data/staminadata/1_training.txt.dat`

See the .ini files for more information, and the --help flag for a short description of the options.

### Input files ###

The default input is formated following the Abadingo formating:

```
num_samples alphabet_size
label length sym1 sym2 ... symN
.
.
.
```
for each symbol, additional data can be attached via /, i.e. `label length sym1/data1 sym2/data2 ... symN/dataN`. These can represent outputs (e.g. for Mealy or Moore machines), or any other information needed by a custom evaluation function.

Real-valued attributes, e.g. for real-time automata, can be attached via :, i.e. `label length sym1:real1,real2,realn ...`. The number of attributes has to be specified in the header after the alphabet size, i.e. `num_samples alphabet_size:num_attributes`.

### Output files ###

flexfringe will generate several .dot files into the specified output directory (./ by default):

*  pre\:\*.dot are intermediary dot files created during the merges/search process.
*  dfafinal.dot is the end result as a dot file
*  dfafinal.dot.json is the end result

You can plot the dot files via

`$ dot -Tpdf file.dot -o outfile.pdf`
or
`$ ./show.sh final.dot`

after installing dot from graphviz.
To use the generated models for language acceptance testing or as a distribution function, it is best to parse the JSON file. You can find an exmaple in the Jupyter notebook at https://github.com/laxris/flexfringe-colab.

## Documentation ##

*flexfringe* has partial Doxygen-style documentation included in the *./doc* directory. It can be regenerated using the settings in Doxygen file.

## Contribution guidelines ##

*  Fork and implement, request pulls.
*  You can find sample evaluation files in ./source/evaluation. Make sure to REGISTER your own file to be able to access it via the -h and --heuristic-name flag.

### Writing tests ###

Unit tests are incomplete. *flexfringe* uses the Catch2 framework (see the [https://github.com/catchorg/Catch2/blob/master/docs/tutorial.md](Tutorial) and the *tests* folder for some examples.

### Logging ###
Logging is incomplete. *flexfringe* uses the loguru framework (see the [https://github.com/emilk/loguru/blob/master/README.md](Loguru documentation)). *flexfringe* uses the stream-version. Please log using the `LOG_S(LEVEL) << "messge"` syntax to implement logging.

## Who to talk to ##

*  Sofia Tsoni (scientific programmer, maintainer)
*  Christian Hammerschmidt (author of the online/streaming mode, interactive mode, and the flexible evaluation function mechanism)
*  Sicco Verwer (original author; best to reach out to for questions on batch mode, RTI+ implementation, and SAT reduction)

## Badges ##
[![Codacy Badge](https://app.codacy.com/project/badge/Grade/2f63a8167ec14bbe8122c3432b3ccfd5)](https://www.codacy.com/bb/chrshmmmr/dfasat/dashboard?utm_source=chrshmmmr@bitbucket.org&amp;utm_medium=referral&amp;utm_content=chrshmmmr/dfasat&amp;utm_campaign=Badge_Grade)

## Credits and Licences ##

*flexfinge* relies on a number of open source packages and libraries. You can find the respective LICENCE files in the source/utility subdirectory.
Most notable, we use

*  CLI11 for command line parsing
*  Catch for unit testing
*  StatsLib C++ and GCE-Math C++ library by Keith O'Hara (Apache Version 2.0)
*  JSON for Modern C++ (version 3.1.2) by Niels Lohmann <http://nlohmann.me> from https://github.com/nlohmann/json

## Documentation

The documentation of this project can be build using the

COMPILE_DOCS=ON

flag along with the cmake command. We are using Doxygen and Sphinx. Requirements for compiling the documentation
are

* Doxygen (tested with version 1.8.20)
* Sphinx (tested with version 3.3.1). We are using the rtd-theme, installation see below.
* breathe (tested with version 4.24.0)

They can be installed on Linux using the commands

```
apt-get install doxygen
```
,

```
pip install sphinx_rtd_theme
```

, and

```
pip install breathe
```
.

IMPORTANT: In case new classes, functions, structs etc. have been added, and they shall show up in the documentation,
they have to be added at the bottom of the docs/index.rst file. For further information and a small quickstart-guide,
please look at e.g.

https://breathe.readthedocs.io/en/latest/quickstart.html
https://breathe.readthedocs.io/en/latest/directives.html

# Flexfringe [![Codacy Badge](https://api.codacy.com/project/badge/Grade/ae975ed72f9c4e1bb19b18dc44aacf1f)](https://app.codacy.com/gh/tudelft-cda-lab/FlexFringe?utm_source=github.com&utm_medium=referral&utm_content=tudelft-cda-lab/FlexFringe&utm_campaign=Badge_Grade_Settings)


Flexfringe, formerly DFASAT, is a flexible state-merging framework written in C++. It consists of a core state merging framework, several different merging routines, a predict module, and an active learning module.

# Install

## Docker

For a cross-platform solution you can use FlexFringe with Docker:

```
docker run -it ghcr.io/tudelft-cda-lab/flexfringe:main
```

This provides a shell inside a docker container at the source repository with the binary included in the source folder. All dependencies to build FlexFringe are included in the container as well.

## Compile

Flexfringe uses CMake as a build tool.

We tested the toolchains on Linux (Debian 12+), MacOS (10.14), and Windows 10.

## Linux and MacOS

You can build and compile the flexfringe project by running in the main directory:

`$ mkdir build && cd build && cmake -DENABLE_PYTHON=On ..`

`$ make`

This builds the executable named *flexfringe*.

Only run the last command in the build folder during development when you change the code.

### Troubleshooting on MacOSX

#### CMake does not find the C- and CXX-compilers
You need to install XCode. Additionally, you need to accept XCode's license. You'll be prompted to accept the license when you try to run it or a module that gets installed.

#### Python packages cannot be found

In this case likely either at least one of the Python_LIBRARIES or Python_INCLUDE_DIRS variables is empty. In this case you need to set them 
manually. 

1. Find out which Python version is used (see CMake output)
2. Find out where this Python instance finds its includes (e.g. via printing the sys.path variable from within the interpreter)
3. The PYTHON_INCLUDE_DIRS variable needs to point to the directory where Python.h resides. PYTHON_LIBRARIES will likely need to point to libpython.[version_number].so or libpython.[version_number].dylib. However, PYTHON_LIBRARIES has always been found in our settings, therefore we cannot comfirm to cover all possible outcomes with this.

Alternatively, you can also simply disable the Python interface, in which case you won't be able to use it (used e.g. for neural network inference). You can disable it via

`$ cmake -DENABLE_PYTHON=Off ..`

from within the build directory.

## Windows

### MSYS2 with CLion

1. Install MSYS2 https://www.msys2.org/docs/installer/

2. Install clang toolchain in the msys2 terminal.

`$ pacman -S mingw-w64-clang-x86_64-toolchain`

and optionally a debugger

`$ pacman -S mingw-w64-clang-x86_64-gdb`

3. Configure a MSYS2 CLion toolchain to use this in *Settings | Build, Execution, Deployment | Toolchains*

4. Set the paths C:\msys64\clang64\bin\clang.exe to the C Compiler and C:\msys64\clang64\bin\clang++.exe to the C++ compiler.

Reference: https://www.jetbrains.com/help/clion/quick-tutorial-on-configuring-clion-on-windows.html#setup-clang

### Windows Visual Studio

1. Install Windows Visual Studio (NOT Visual Studio Code) and include CMake during install: https://learn.microsoft.com/nl-nl/cpp/build/cmake-projects-in-visual-studio?view=msvc-170&viewFallbackFrom=vs-2019

2. Open the *Developer Command Prompt*

3. Run the following:

`$ mkdir build`

`$ cmake -S . -B build`

`$ cmake --build build --target flexfringe`

The executable is now available in `build\Debug\flexfringe.exe`.

Only run the last command during development when you change the code.

This should also be possible to be done/configured from an IDE such as Visual Studio itself, CLion, or VS Code, etc.

## Additionally features

### Database (Active Learning)

Database support is available behind the feature flag `ENABLE_DATABASE`. Enable during cmake configuration with `-DENABLE_DATABASE`.

On Debian-based systems install dependencies with `sudo apt-get install libpq-dev` and `sudo apt-get install postgres` if you also want to actually use the database.

On MacOS install with Homebrew using `brew install libpq` and `brew install postgres` for the full database package. Alternatively install with MacPorts using `port install postgresxx` where xx is the version number of the postgresql version you want. 

On Windows install from: https://www.postgresql.org/download/

### Deep Learning (Active Learning)

Deep learning is done with python integration. This support is available behind the feature flag `ENABLE_PYTHON`. Enable during cmake configuration with `-DENABLE_PYTHON`.

On Debian-based systems install dependencies with `sudo apt-get install python3-dev`.

On MacOS install with Homebrew using `brew install python3` or MacPorts using `brew install python3xx` where xx is again the desired version number.

On Windows install from: https://www.python.org/downloads/

NB. On Windows the Python extension could not be successfully linked with the MSVC toolchain the last time this was checked. Use the MSYS2 toolchain in CLion instead.

### SAT Solver (Passive Learning)

For expert users: In case you want to use the reduction to SAT and automatically invoke the SAT solver, you need to provide the path to the solver binary. flexfringe has been tested with lingeling (which you can get from http://fmv.jku.at/lingeling/ and run its build.sh).
**PLEASE NOTE:** SAT solving only works for learning plain DFAs. The current implementation is not verified to be correct. Use an older commit if you rely on SAT-solving.

# Usage

Run `./flexfringe --help` to get help.

We provide several `.ini` files as a shortcut to storing commonly used settings.

Example:

`$ ./flexfringe --ini ini/batch-overlap.ini data/staminadata/1_training.txt.dat`

See the `.ini` files for more information, and the `--help` flag for a short description of the options.

## Input files

The default input is formated following the Abbadingo formating:

```
num_samples alphabet_size
label length sym1 sym2 ... symN
.
.
.
```
for each symbol, additional data can be attached via `/`, i.e. `label length sym1/data1 sym2/data2 ... symN/dataN`. These can represent outputs (e.g. for Mealy or Moore machines), or any other information needed by a custom evaluation function.

Real-valued attributes, e.g. for real-time automata, can be attached via `:`, i.e. `label length sym1:real1,real2,realn ...`. The number of attributes has to be specified in the header after the alphabet size, i.e. `num_samples alphabet_size:num_attributes`.

## Output files

flexfringe will generate several .dot files into the specified output directory (./ by default):

*  ``pre:*.dot`` are intermediary dot files created during the merges/search process.
*  `dfafinal.dot` is the end result as a dot file
*  `dfafinal.dot.json` is the end result

You can plot the dot files via

`$ dot -Tpdf file.dot -o outfile.pdf`

or

`$ ./show.sh final.dot`

after installing dot from graphviz.
To use the generated models for language acceptance testing or as a distribution function, it is best to parse the JSON file. You can find an exmaple in the Jupyter notebook at https://github.com/laxris/flexfringe-colab.

## Documentation ##

*flexfringe* has partial Doxygen-style documentation included in the *./doc* directory. It can be regenerated using the settings in Doxygen file.

# Contribute

*  Fork and implement, request pulls.
*  You can find sample evaluation files in ./source/evaluation. Make sure to REGISTER your own file to be able to access it via the -h and --heuristic-name flag.

## Writing tests

Unit tests are incomplete. *flexfringe* uses the Catch2 framework (see the [https://github.com/catchorg/Catch2/blob/master/docs/tutorial.md](Tutorial) and the *tests* folder for some examples.

## Logging
Logging is incomplete. *flexfringe* uses the loguru framework (see the [https://github.com/emilk/loguru/blob/master/README.md](Loguru documentation)). *flexfringe* uses the stream-version. Please log using the `LOG_S(LEVEL) << "message"` syntax to implement logging.

## Who to talk to

*   Christian Hammerschmidt (author of the online/streaming mode, interactive mode, and the flexible evaluation function mechanism)
*   Sicco Verwer (original author; best to reach out to for questions on batch mode, RTI+ implementation, and SAT reduction)
*   Robert Baumgartner (Former PhD student, author of the streaming mode and the active learning module)
*   Hielke Walinga (Former Master student, author of the database connector)

Former contributors include:
*   Tom Catshoek (scientific programmer and maintainer, wrote the Lexy-based parser)
*   Sofia Tsoni (formerly scientific programmer and maintainer)

# Credits and Licences

*flexfinge* relies on a number of open source packages and libraries. You can find the respective LICENCE files in the source/utility subdirectory.
Most notable, we use

*   CLI11 for command line parsing
*   Catch for unit testing
*   StatsLib C++ and GCE-Math C++ library by Keith O'Hara (Apache Version 2.0)
*   JSON for Modern C++ (version 3.1.2) by Niels Lohmann <http://nlohmann.me> from https://github.com/nlohmann/json
*   Lexy for formal parsing (https://lexy.foonathan.net/)
*   Libpqxx for the database connecting (https://lexy.foonathan.net/)
*   Fmt from https://github.com/fmtlib/fmt

# Building Doxygen Documentation

TODO:
The documentation of this project can be build using the

```
COMPILE_DOCS=ON
``````

flag along with the cmake command. We are using Doxygen and Sphinx. Requirements for compiling the documentation
are

*   Doxygen (tested with version 1.8.20)
*   Sphinx (tested with version 3.3.1). We are using the rtd-theme, installation see below.
*   breathe (tested with version 4.24.0)

They can be installed on Linux using the commands

```shell
apt-get install doxygen
pip install sphinx_rtd_theme
pip install breathe
```

IMPORTANT: In case new classes, functions, structs etc. have been added, and they shall show up in the documentation,
they have to be added at the bottom of the docs/index.rst file. For further information and a small quickstart-guide,
please look at e.g.

https://breathe.readthedocs.io/en/latest/quickstart.html
https://breathe.readthedocs.io/en/latest/directives.html

# Cite

## Core
* https://ieeexplore.ieee.org/abstract/document/8094471
* https://arxiv.org/abs/2203.16331

## Streaming
* https://arxiv.org/abs/2207.01516
* https://proceedings.mlr.press/v217/baumgartner23a.html

## Active Learning
* https://arxiv.org/abs/2406.18328
* https://link.springer.com/chapter/10.1007/978-3-031-71112-1_4

## Databases
* https://arxiv.org/abs/2406.07208
* https://repository.tudelft.nl/record/uuid:40a1cb17-a46e-421e-a916-396ebaf3d7b1

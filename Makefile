CC	=	g++
CFLAGS	=	-O4
SOURCES = 	source/*.cpp 
LFLAGS 	= -w -std=c++17 -L/opt/local/lib -I/opt/local/include -I./source -I./source/evaluation -I./source/utility -lm -lpthread -ldl

EVALFILES := $(wildcard source/evaluation/*.cpp)
EVALOBJS := $(addprefix source/evaluation/,$(notdir $(EVALFILES:.cpp=.o)))

EVALFILES := $(filter-out $(PYTHON_EVAL), $(EVALFILES))
EVALOBJS := $(addprefix source/evaluation/,$(notdir $(EVALFILES:.cpp=.o)))

OUTDIR ?= .

.PHONY: all clean test

all: regen source/gitversion.cpp flexfringe

regen:
	sh collector.sh

debug:
	$(CC) -g $(SOURCES) -o flexfringe $(LFLAGS) $(LIBS)

flexfringe: $(SOURCES) $(EVALOBJS) source/gitversion.cpp
	$(CC) $(CFLAGS) -o $@ -DLOGURU_WITH_STREAMS=1 source/utility/loguru.cpp $(SOURCES)  $(EVALOBJS) -I./ $(LFLAGS) $(LIBS)


test: regen $(EVALOBJS) source/gitversion.cpp
	$(CC) $(FLAGS) -DLOGURU_WITH_STREAMS=1 -DUNIT_TESTING=1 source/utility/loguru.cpp -I./ -o runtests tests/tests.cpp tests/tail.cpp $(SOURCES) $(EVALOBJS) $(LFLAGS) $(LIBS)
	mkdir -p test-reports
	./runtests -r junit > test-reports/testresults.xml	

source/evaluation/%.o: source/evaluation/%.cpp
	$(CC) $(CFLAGS) -fPIC -c -DLOGURU_WITH_STREAMS=1 -o $@ $< -I.source $(LFLAGS) $(LIBS) $(PYTHON_INC) $(PYTHON_LIBS) $(BOOST_LIBS) 

clean:
	rm -f flexfringe ./source/evaluation/*.o source/generated.cpp named_tuple.py *.dot *.json exposed_decl.pypp.txt flexfringe*.so source/gitversion.cpp source/evaluators.h

source/gitversion.cpp: 
	[ -e .git/HEAD ] && [ -e .git/index ] && echo "const char *gitversion = \"$(shell git rev-parse HEAD)\";" > $@ || echo "const char *gitversion = \"No commit info available\";" > $@


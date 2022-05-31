#!/bin/bash

g++ -std=c++11 -DUNIT_TESTING=1 -Wall -I../source/utility -I/home/chris/Dropbox/dfasat/source/evaluation -I/home/chris/Dropbox/dfasat/source/ -I/home/chris/Dropbox/dfasat -I/home/chris/Dropbox/dfasat/source/utility -o tests tests.cpp tail.cpp ../source/*.cpp ../source/evaluation/*.o -lm -lpopt -lgsl -lgslcblas -lpthread -ldl

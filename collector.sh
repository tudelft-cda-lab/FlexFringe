#!/bin/bash

HEADER=source/evaluators.h
rm $HEADER
echo "#ifndef __ALL_HEADERS__" > $HEADER
echo "#define __ALL_HEADERS__" >> $HEADER
for file in $(ls source/evaluation/*.h)
do
	echo "#include \"evaluation/$(basename "$file")\"" >> $HEADER
done
echo "#endif" >> $HEADER

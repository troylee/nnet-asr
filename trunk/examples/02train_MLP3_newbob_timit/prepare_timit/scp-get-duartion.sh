#!/bin/bash

for fileL in $*; do
  for file in $(cat $fileL); do
    samples=$(HList -z -h $file | grep Samples | awk '{ print $3 }')
    file=${file##*/}; file=${file%.*}
    echo $file $samples
  done
done
 



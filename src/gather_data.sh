#!/bin/bash

for mask in thrust_cpu vexcl_cpu viennacl_cpu \
    thrust_gpu vexcl_1gpu viennacl_gpu \
    vexcl_2gpu vexcl_3gpu
do
    cat $mask_*.dat | sort -g > $mask.dat
done

#!/bin/bash

for mask in thrust_cpu vexcl_cpu_intel vexcl_cpu_amd \
    viennacl_cpu_intel viennacl_cpu_amd \
    thrust_gpu vexcl_1gpu viennacl_gpu \
    vexcl_2gpu vexcl_3gpu
do
    cat ${mask}_*.dat | sort -g > ${mask}.dat
done

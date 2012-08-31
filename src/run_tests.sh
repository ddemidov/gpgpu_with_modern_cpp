#!/bin/bash

for exp in damped_oscillator disordered_ham_lattice \
    lorenz_ensemble phase_oscillator_chain
do
    pushd $exp
    for job in *.sge; do
	qsub $job
    done
    popd
done

#!/bin/bash

for exp in damped_oscillator disordered_ham_lattice \
    lorenz_ensemble phase_oscillator_chain
do
    pushd $exp
    for job in cpu 1gpu 2gpu 3gpu; do
	qsub run_on_${job}.sge
    done
    popd
done

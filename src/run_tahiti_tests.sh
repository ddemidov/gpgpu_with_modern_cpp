#!/bin/bash
for exp in damped_oscillator disordered_ham_lattice \
        lorenz_ensemble phase_oscillator_chain
do
    pushd $exp
    ./run_on_tahiti.sh
    popd
done

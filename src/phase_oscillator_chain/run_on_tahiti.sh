#!/bin/bash

export OCL_DEVICE=Tahiti

# warming runs
./vexcl_phase_oscillator
./viennacl_phase_oscillator

rm -f vexcl_1gpu.dat
rm -f vexcl_2gpu.dat
rm -f vexcl_3gpu.dat
rm -f viennacl_gpu.dat

for ((a=256;a<=4194304;a*=2)); do
    echo "$a "

    for ((task=1;task<=10;task++)); do
	echo "  $task"

	for ((ndev=1;ndev<=3;ndev++)); do
	    echo "    vexcl ${ndev}"
	    export OCL_MAX_DEVICES=${ndev}
	    echo -n "$a " >> vexcl_${ndev}gpu.dat
	    /usr/bin/time -f %e -o vexcl_${ndev}gpu.dat -a ./vexcl_phase_oscillator $a > /dev/null
	done

	echo "    viennacl"
	echo -n "$a " >> viennacl_gpu.dat
	/usr/bin/time -f %e -o viennacl_gpu.dat -a ./viennacl_phase_oscillator $a > /dev/null
    done
done

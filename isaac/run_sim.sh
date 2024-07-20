#!/bin/bash

f="sim_on_test_obj2hand"
command_prefix="python isaac/simulate_test.py --cfg experiments/ugg_simulation_test.yaml"
total=1125
stp=25

i=0
while [ $i -le $total ]; do
    full_command="$command_prefix --folder=$f --start=$i --step=$stp"

    echo "Running command: $full_command"

    eval $full_command

    i=$((i + stp))
done

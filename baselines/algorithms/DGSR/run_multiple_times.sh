#!/bin/bash
num_executions=20

for i in $(seq 1 $num_executions)
do
    echo "run $i started"
    taskset -c 70-80 python run_test_DGSR_customdata.py -s $i
    echo "run $i finished"
    echo "-------------------"
    sleep 1
done

echo "all runs finished"

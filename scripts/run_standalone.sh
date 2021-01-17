#!/bin/bash

SOURCE_PATH=$1
TARGET_PATH=$2

CFG=$3

set -e

for SEED in 2 3 4
do
OUTPUT=$TARGET_PATH/seed$SEED
INPUT=$SOURCE_PATH/seed-1/run-$SEED/parameters-log-0.npy
mkdir -p $OUTPUT
python scripts/standalone_run_obj_perturb.py --config_path $CFG --outputfile=$OUTPUT --parameter_path=$INPUT --carla_port 2000
done
#!/bin/bash
#SBATCH --job-name=import-test       # name
#SBATCH --nodes=2                    # num nodes
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task 1
#SBATCH --output=%x-%j.out

set -e

echo "START TIME: $(date)"

################# ARGUMENTS #################
############################################# 

# Distributed launcher to use. Currently supports from ["accelerate", "torchrun"]
export LAUNCHER=${LAUNCHER:-"accelerate"}

# Number of tests to run. We need multiple to reproduce due to stochasticity
export NUM_TESTS=${NUM_TESTS:-100}

# The image to use. Sif path for apptainer, image name for docker
export IMAGE=${IMAGE:-"import_test.sif"}

# The python script to run. Can be changed if you build with a different test script
export PROGRAM=${PROGRAM:-"/app/test_imports.py"}

# Binds for container with appropriate option formatting
# e.g. for apptainer: BINDS="--bind --bind ${TEST_SCRIPT_PATH}:${PROGRAM}"
export BINDS=${BINDS:-""}

# If not "true", uses Docker instead
export USE_APPTAINER=${USE_APPTAINER:-"true"}

# If import error occurs the program hangs, so we need a timeout (in seconds)
export TIMEOUT=${TIMEOUT:-60}

# If non-empty, prints out the entire python import trace
export PYTHONVERBOSE=${PYTHONVERBOSE:-""}

# If true, separates the outputs of each node into separate files
export SPLIT_OUTPUT=${SPLIT_OUTPUT:-true}

############################################# 
############################################# 

# CHANGE TO CUMMULATIVELY LOG OUTPUTS
LOG_PATH="main_log.txt"

# For this test we don't actually use GPUs, but we use this to launch the appropriate number of processes
export GPUS_PER_NODE=8
export NNODES=$SLURM_NNODES
export NUM_PROCESSES=$(expr $NNODES \* $GPUS_PER_NODE)

# so processes know who to talk to
nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}

# Needed for the nodes to communicate
export MASTER_ADDR=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname -I | awk '{print $1}')

export MASTER_PORT=29500

# Determine container command
if [[ $USE_APPTAINER == "true" ]]; then
    APPTAINER_OPTIONS="--nv "
    CONTAINER_CMD="apptainer run ${APPTAINER_OPTIONS} ${BINDS} ${IMAGE} "
else
    echo "USING DOCKER"
    DOCKER_OPTIONS="--gpus all --network host --shm-size=512m "

    # Docker pull can take time if not cached locally. Need to pull and wait so we don't timeout later
    # WARNING: This assumes the nodes share the same filesystem for docker
    srun docker pull $IMAGE
    wait

    CONTAINER_CMD="docker run ${DOCKER_OPTIONS} ${BINDS} ${IMAGE}"
fi

# OTHER LAUNCHERS CAN BE USED HERE
# Note: it is important to escape `$SLURM_PROCID` since we want the srun on each node to evaluate this variable

if [[ $LAUNCHER == "accelerate" ]]; then
    export LAUNCHER_CMD="accelerate launch \
        --main_process_ip $MASTER_ADDR \
        --main_process_port $MASTER_PORT \
        --machine_rank \$SLURM_PROCID \
        --num_processes $NUM_PROCESSES \
        --num_machines $NNODES \
        $PROGRAM"
elif [[ $LAUNCHER == "torchrun" ]]; then
    export LAUNCHER_CMD="torchrun
        --nproc_per_node ${GPUS_PER_NODE} \
        --nnodes ${NNODES} \
        --node_rank \$SLURM_PROCID \
        --master_addr ${MASTER_ADDR} \
        --master_port ${MASTER_PORT}\
        $PROGRAM"
elif [[ $LAUNCHER == "deepspeed" ]]; then
    # TODO:
    #   * Build hostfile
    #   * Launch on ONLY the primary node using deepspeed
    exit 1
elif [[ $LAUNCHER == "python" ]]; then
    # If not running with a launcher, we need to build the processes ourselves
    # TODO: fix the command below to not have a syntax error
    export MAX_LOCAL_RANK=$((GPUS_PER_NODE - 1))
    echo "MAX_LOCAL_RANK: $MAX_LOCAL_RANK"
    export LAUNCHER_CMD="bash -c '\
        for i in \$(seq 0 $MAX_LOCAL_RANK); do \
            export LOCAL_RANK=\${i} ; \
            python3 $PROGRAM ; \
        done'"
else
    echo "Launcher $LAUNCHER not supported"
    exit 1
fi

export CMD="$CONTAINER_CMD $LAUNCHER_CMD"

echo "STARTING TEST WITH CMD ${CMD}"

SLURM_OPTIONS="--jobid $SLURM_JOBID "
if [[ $SPLIT_OUTPUT == "true" ]]; then
    SLURM_OPTIONS="${SLURM_OPTIONS} --output %x-%j-%t.out --err %x-%j-%t.out --open-mode append "
fi

for i in `seq 1 $NUM_TESTS`
do
    echo ">>>> Test $i"
    srun ${SLURM_OPTIONS} timeout ${TIMEOUT} bash -c "$CMD" 2>&1 | tee -a $LOG_PATH
    # TODO: Check exit code if failure, echo to signify as such, and increment a counter to echo at the end
    # For now to determine the number of failures, we need grep the log files
done

echo "END TIME: $(date)"
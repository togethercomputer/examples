#!/bin/bash
#SBATCH --nodes 2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=8
#SBATCH --exclusive
#SBATCH -o slurm-nccl-%j.out

# This script will spin up NUM_NODES * GPUS_PER_NODE tasks, each claiming a gpu on the node and
# synchronizing using a file in WORK_DIR. This is used to test the interconnect between GPUs and nodes
# For different node configurations, change the sbatch options above

while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --work-dir)
            # WORK_DIR must be in a SHARED directory since it will be used 
            # to initialize communication
            export WORK_DIR="$2"
            shift
            shift
            ;;
        --image)
            # APPTAINER_IMAGE is recommended to be in a shared directory
            # but must at least exist in the same path for all ndoes
            export APPTAINER_IMAGE="$2"
            shift
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

if [ -z "${WORK_DIR}" ]; then
  echo "Error: WORK_DIR environment variable is not set"
  exit 1
fi
if [ -z "${APPTAINER_IMAGE}" ]; then
  echo "Error: APPTAINER_IMAGE environment variable is not set"
  exit 1
fi

mkdir -p $WORK_DIR
# Need higher debug for checking on IB
export NCCL_DEBUG=INFO

# We bind /etc/nccl.conf and /etc/crusoe for nccl env vars and topolyg
# Unbind if running on a non Forge configuration
srun apptainer run \
    --nv \
    --bind $WORK_DIR:/scratch \
    --bind /etc/nccl.conf:/etc/nccl.conf \
    --bind /etc/crusoe:/etc/crusoe \
    --env SCRATCH=/scratch \
    $APPTAINER_IMAGE python3 /app/tests/test_nccl.py
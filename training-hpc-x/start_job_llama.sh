#!/bin/bash
#SBATCH --nodes=2
#SBATCH --job-name=ai-multi-gpu
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH --gres=gpu:8

export TORCH_DTYPE=bf16

export TRAIN_OPTIONS=""
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --work-dir)
            # WORK_DIR is binded to the apptainer as /work and is used for model outputs.
            # To specify dataset files in --train-options, all files must be in WORK_DIR,
            # and specified as "/work/<PATH FROM WORK_DIR>"
            # Additionally, HF transformers and datasets cache will be $WORK_DIR/cache
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
        --fp16)
            # Optimizer with FSDP offloading only tested with bf16
            echo  "WARNING: FP16 WITH OFFLOADING NOT TESTED!"
            # If specified, will use bfloat16 instead of fp16
            # Do not use the the dtype options on --train-options
            export TORCH_DTYPE="fp16"
            shift
            ;;
        --train-options)
          export TRAIN_OPTIONS="$2"
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

mkdir -p $WORK_DIR/output

# Without this, srun does not inherit cpus-per-task from sbatch.
export SRUN_CPUS_PER_TASK="$SLURM_CPUS_PER_TASK"

# so processes know who to talk to
MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"

echo "MASTER ADDR: $MASTER_ADDR"
# Get IP for hostname.
export MASTER_ADDR="$(nslookup "$MASTER_ADDR" | grep -oP '(?<=Address: ).*')"
echo "MASTER ADDR (IP): $MASTER_ADDR"

export MASTER_PORT=7010

export GPUS_PER_NODE=8

# Set up accelerate config.
export ACCELERATE_CONFIG_YAML=$WORK_DIR/accelerate_config.yaml

srun bash -c "cat <<EOT > \"\$ACCELERATE_CONFIG_YAML\"
compute_environment: LOCAL_MACHINE
distributed_type: FSDP
gpu_ids: all
fsdp_config:
  fsdp_auto_wrap_policy: TRANSFORMER_BASED_WRAP
  fsdp_backward_prefetch_policy: BACKWARD_PRE
  fsdp_offload_params: true
  fsdp_sharding_strategy: 1
  fsdp_state_dict_type: FULL_STATE_DICT
  fsdp_transformer_layer_cls_to_wrap: LlamaDecoderLayer
main_process_ip: '\$MASTER_ADDR'
main_process_port: \$MASTER_PORT
main_training_function: main
mixed_precision: \$TORCH_DTYPE
num_machines: \$SLURM_JOB_NUM_NODES
num_processes: \$((SLURM_JOB_NUM_NODES * GPUS_PER_NODE))
rdzv_backend: c10d
same_network: true
EOT"

echo "ACCELERATE CONFIG:"
cat $ACCELERATE_CONFIG_YAML

# Run the demo.
srun . /opt/hpcx/hpcx-init.sh && \
  hpcx_load && \
  bash -c 'apptainer run \
  --nv \
  --bind /etc/nccl.conf:/etc/nccl.conf \
  --bind /etc/crusoe:/etc/crusoe \
  --bind $WORK_DIR:/work:rw \
  --env TRANSFORMERS_CACHE=/work/cache/transformers \
  --env HF_DATASETS_CACHE=/work/cache/datasets \
  $APPTAINER_IMAGE accelerate launch \
  --machine_rank=$SLURM_NODEID \
  --config_file=/work/accelerate_config.yaml \
  /app/train_causal_lm.py $TRAIN_OPTIONS --torch-dtype $TORCH_DTYPE --output-dir /work/output'
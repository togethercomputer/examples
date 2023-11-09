#!/bin/bash
#SBATCH --nodes=2
#SBATCH --job-name=ai-multi-gpu
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH --gres=gpu:8

export APPTAINER_IMAGE=${APPTAINER_IMAGE:-"./training_ex.sif"}

# Without this, srun does not inherit cpus-per-task from sbatch.
export SRUN_CPUS_PER_TASK="$SLURM_CPUS_PER_TASK"

# so processes know who to talk to
MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
# Allow communication over InfiniBand cells.
# MASTER_ADDR="${MASTER_ADDR}i"
echo "MASTER ADDR: $MASTER_ADDR"
# Get IP for hostname.
export MASTER_ADDR="$(nslookup "$MASTER_ADDR" | grep -oP '(?<=Address: ).*')"
echo "MASTER ADDR (IP): $MASTER_ADDR"

export MASTER_PORT=7010

export GPUS_PER_NODE=8

# Set up accelerate config.
export ACCELERATE_CONFIG_YAML=accelerate_config.yaml

srun bash -c "((\$SLURM_PROCID)) || cat <<EOT > \"\$ACCELERATE_CONFIG_YAML\"
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
mixed_precision: bf16
num_machines: \$SLURM_JOB_NUM_NODES
num_processes: \$((SLURM_JOB_NUM_NODES * GPUS_PER_NODE))
rdzv_backend: c10d
same_network: true
EOT"

echo "ACCELERATE CONFIG:"
cat $ACCELERATE_CONFIG_YAML

# Run the demo.
srun bash -c 'apptainer run \
  --nv \
  --bind $ACCELERATE_CONFIG_YAML:/mnt/config.yaml \
  --bind /etc/nccl.conf:/etc/nccl.conf \
  --bind /etc/crusoe:/etc/crusoe \
  $APPTAINER_IMAGE accelerate launch \
  --machine_rank=$SLURM_NODEID \
  --config_file=/mnt/config.yaml \
  /app/train.py --batch-size 1'
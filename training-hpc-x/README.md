# Training with Infiniband (IB)

This example will walk through a sample of fine-tuning Llama-2-7b with an Infiniband multi-node slurm configuration on your Together Forge cluster. In our case we will use HuggingFace Accelerate with Fully Sharded Data Parallelism and Flash Attention V2.

## Setting Up the Container

A generic build of PyTorch is not guaranteed to work with every IB fabric. NVIDIA's HPC-X can fix this problem, but this requires building PyTorch from source against HPC-X. Luckily NVIDIA provides a [prebuilt container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch) for this purpose. The Dockerfile for this project adds dependencies such as Accelerate, Transformers and Flash Attention V2.

You can build a Singularity Image Format file from the DockerHub repository to run the image with Apptainer (make sure to specify `APPTAINER_IMAGE`)
```console
apptainer build $APPTAINER_IMAGE docker://togethercomputer/infiniband_training_example:latest
```

## Testing NCCL

Before training, let's make sure PyTorch is able to use NCCL with RDMA. First, enable NCCL debug logging
```console
export NCCL_DEBUG=INFO
```
When launching any command with a container, it is **IMPORTANT** we include the appropriate NCCL configs, plugins and topology information. You can do this by adding the following bind to any `apptainer run` command. We take care of this for you in the bash scripts of this example.
```console
    --bind /etc/nccl.conf:/etc/nccl.conf
    --bind /etc/crusoe:/etc/crusoe 
```


Now we can test NCCL to make sure it's using RDMA interfaces and not RoCE. Torch Distributed will require a file in a shared directory to initialize communications for this test, which we specify with `WORK_DIR`. Simply run our test script with `NCCL_DEBUG=INFO` and inspect the logs. This will use two nodes and test communication between one gpu on each node.

```console
sbatch tests/test_nccl.sh --work-dir $WORK_DIR --image $APPTAINER_IMAGE 
```

You should see the following in the logs to verify we are using RDMA with the correct configs and topology.
```console
NCCL INFO NET/Plugin: Loaded net plugin NCCL RDMA Plugin v6 (v6)

NCCL INFO Plugin Path : /opt/hpcx/nccl_rdma_sharp_plugin/lib/libnccl-net.so

NCCL INFO NCCL_TOPO_FILE set by environment to /etc/crusoe/nccl_topo/h100-80gb-sxm-ib.xml
```

If misconfigured, NCCL may default to RoCE. If this happens, verify the binds from before. Additionally, if you are using a containerization other than Apptainer (e.g. Docker), make sure the shared memory is at least 1GB.


## Launch Training Over Infiniband

If NCCL is able to correctly use RDMA, most distributed training frameworks are able to make use of Infiniband without additional setup. Just make sure your training script of choice is using an NCCL backend for communication.

In this example, we use HuggingFace's Accelerate library to launch a distributed fine-tuning job of Llama-2-7b using FSDP on two nodes with the [togethercomputer/llama-instruct datset](https://huggingface.co/datasets/togethercomputer/llama-instruct). Make sure you have the `HUGGING_FACE_HUB_TOKEN` environment variable set to an API key with access to the model. The main process (global rank 0 of 16) synchronizes with other processes through an exposed port. If using Apptainer, all ports are exposed from the container by default, so it's once again as simple as queuing a job. Note that for FSDP, batch-size is specified per-node.

```console
sbatch start_job_llama.sh \
    --work-dir $WORK_DIR \
    --image $APPTAINER_IMAGE \
    --train-options \
        "--batch-size 1 \
        --use-flash-attn \
        --train-file /work/jokes.jsonl"
```

If using `NCCL_DEBUG=INFO`, you should be able to validate that Infiniband is being used as before. A shared filesystem is not required for this script. You will need access to the [Llama-2 HuggingFace Repository](https://huggingface.co/meta-llama/Llama-2-7b-hf) to immediately begin training. You can also use another model or dataset (just disable Flash Attention and modify max-length if not supported). The options below can be specified in a string passed `-train-options`.

```console
-h, --help            show this help message and exit
  --model-name-or-path MODEL_NAME_OR_PATH
                        model path
  --train-file TRAIN_FILE
                        Train dataset file (jsonl) or huggingface hub datset
  --valid-file VALID_FILE
                        Validation dataset file (jsonl) or huggingface hub datset. If None, uses a split of train.
  --valid-split VALID_SPLIT
                        Percent of train to split in to validation if dedicated validation file not specified.
  --batch-size BATCH_SIZE
                        Batch Size
  --text-column TEXT_COLUMN
                        Column for the text in the datsets
  --max-length MAX_LENGTH
                        Max sequence length.
  --learning-rate LEARNING_RATE
                        Learning Rate
  --num-epochs NUM_EPOCHS
                        Number of epochs
  --use-flash-attn      Use flash attention v2 (not supported by all models)
  --torch-dtype {bf16,fp16,fp32}
                        Torch data type
  --output-dir OUTPUT_DIR
                        Output dir to save model
```

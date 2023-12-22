# Building the SIF

You can build using the layers from Docker Hub at `azahed98/import_test`

```console
$ apptainer build import_test.sif docker://azahed98/import_test
```

# Running the test

The provided script uses a SLURM system. To start a job, simply run

```console
$ sbatch test_imports.slurm
```

The configurable environment variables for teh script are provided below.

```bash
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

```

# The resulting error

If reproduced, the script will eventually generate a circular import error that looks like the following

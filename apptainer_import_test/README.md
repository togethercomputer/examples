# Building the SIF

You can build using the layers from Docker Hub at `azahed98/import_test`

```console
$ apptainer build import_test.sif docker://azahed98/import_test
```

# Running the test

The provided script uses a SLURM system with at least 2 nodes for the job. To start a job, simply run

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

If reproduced, the script will eventually generate a circular import error that looks like the following, always originating from transformer import_utils

```python
Traceback (most recent call last):
  File "/usr/local/lib/python3.8/dist-packages/transformers/utils/import_utils.py", line 1386, in _get_module
Traceback (most recent call last):
  File "/usr/local/lib/python3.8/dist-packages/transformers/utils/import_utils.py", line 1386, in _get_module
Traceback (most recent call last):
  File "/usr/local/lib/python3.8/dist-packages/transformers/utils/import_utils.py", line 1386, in _get_module
    return importlib.import_module("." + module_name, self.__name__)
  File "/usr/lib/python3.8/importlib/__init__.py", line 127, in import_module
    return importlib.import_module("." + module_name, self.__name__)
  File "/usr/lib/python3.8/importlib/__init__.py", line 127, in import_module
    return importlib.import_module("." + module_name, self.__name__)
  File "/usr/lib/python3.8/importlib/__init__.py", line 127, in import_module
[2023-12-21 17:17:13,535] torch.distributed._tensor._xla: [WARNING] No module named 'torch_xla'
    return _bootstrap._gcd_import(name[level:], package, level)
  File "<frozen importlib._bootstrap>", line 1014, in _gcd_import
    return _bootstrap._gcd_import(name[level:], package, level)
  File "<frozen importlib._bootstrap>", line 1014, in _gcd_import
    return _bootstrap._gcd_import(name[level:], package, level)
  File "<frozen importlib._bootstrap>", line 1014, in _gcd_import
  File "<frozen importlib._bootstrap>", line 991, in _find_and_load
  File "<frozen importlib._bootstrap>", line 991, in _find_and_load
  File "<frozen importlib._bootstrap>", line 991, in _find_and_load
  File "<frozen importlib._bootstrap>", line 975, in _find_and_load_unlocked
  File "<frozen importlib._bootstrap>", line 671, in _load_unlocked
  File "<frozen importlib._bootstrap_external>", line 848, in exec_module
  File "<frozen importlib._bootstrap>", line 975, in _find_and_load_unlocked
  File "<frozen importlib._bootstrap>", line 671, in _load_unlocked
  File "<frozen importlib._bootstrap_external>", line 848, in exec_module
  File "<frozen importlib._bootstrap>", line 975, in _find_and_load_unlocked
  File "<frozen importlib._bootstrap>", line 671, in _load_unlocked
  File "<frozen importlib._bootstrap_external>", line 848, in exec_module
  File "<frozen importlib._bootstrap>", line 219, in _call_with_frames_removed
  File "/usr/local/lib/python3.8/dist-packages/transformers/trainer.py", line 59, in <module>
  File "<frozen importlib._bootstrap>", line 219, in _call_with_frames_removed
  File "/usr/local/lib/python3.8/dist-packages/transformers/trainer.py", line 59, in <module>
  File "<frozen importlib._bootstrap>", line 219, in _call_with_frames_removed
  File "/usr/local/lib/python3.8/dist-packages/transformers/trainer.py", line 59, in <module>
    from .data.data_collator import DataCollator, DataCollatorWithPadding, default_data_collator
  File "/usr/local/lib/python3.8/dist-packages/transformers/data/__init__.py", line 26, in <module>
    from .data.data_collator import DataCollator, DataCollatorWithPadding, default_data_collator
  File "/usr/local/lib/python3.8/dist-packages/transformers/data/__init__.py", line 26, in <module>
[2023-12-21 17:17:13,541] torch.distributed._tensor._xla: [WARNING] No module named 'torch_xla'
    from .data.data_collator import DataCollator, DataCollatorWithPadding, default_data_collator
  File "/usr/local/lib/python3.8/dist-packages/transformers/data/__init__.py", line 26, in <module>
    from .metrics import glue_compute_metrics, xnli_compute_metrics
  File "/usr/local/lib/python3.8/dist-packages/transformers/data/metrics/__init__.py", line 19, in <module>
    from .metrics import glue_compute_metrics, xnli_compute_metrics
  File "/usr/local/lib/python3.8/dist-packages/transformers/data/metrics/__init__.py", line 19, in <module>
    from .metrics import glue_compute_metrics, xnli_compute_metrics
  File "/usr/local/lib/python3.8/dist-packages/transformers/data/metrics/__init__.py", line 19, in <module>
    from scipy.stats import pearsonr, spearmanr
  File "/usr/local/lib/python3.8/dist-packages/scipy/stats/__init__.py", line 485, in <module>
    from scipy.stats import pearsonr, spearmanr
  File "/usr/local/lib/python3.8/dist-packages/scipy/stats/__init__.py", line 485, in <module>
    from scipy.stats import pearsonr, spearmanr
  File "/usr/local/lib/python3.8/dist-packages/scipy/stats/__init__.py", line 485, in <module>
    from ._stats_py import *
  File "/usr/local/lib/python3.8/dist-packages/scipy/stats/_stats_py.py", line 39, in <module>
    from ._stats_py import *
  File "/usr/local/lib/python3.8/dist-packages/scipy/stats/_stats_py.py", line 39, in <module>
    from ._stats_py import *
  File "/usr/local/lib/python3.8/dist-packages/scipy/stats/_stats_py.py", line 39, in <module>
    from scipy.spatial.distance import cdist
  File "/usr/local/lib/python3.8/dist-packages/scipy/spatial/__init__.py", line 105, in <module>
    from scipy.spatial.distance import cdist
  File "/usr/local/lib/python3.8/dist-packages/scipy/spatial/__init__.py", line 105, in <module>
    from scipy.spatial.distance import cdist
  File "/usr/local/lib/python3.8/dist-packages/scipy/spatial/__init__.py", line 105, in <module>
    from ._kdtree import *
  File "/usr/local/lib/python3.8/dist-packages/scipy/spatial/_kdtree.py", line 4, in <module>
    from ._kdtree import *
  File "/usr/local/lib/python3.8/dist-packages/scipy/spatial/_kdtree.py", line 4, in <module>
    from ._kdtree import *
  File "/usr/local/lib/python3.8/dist-packages/scipy/spatial/_kdtree.py", line 4, in <module>
    from ._ckdtree import cKDTree, cKDTreeNode
  File "_ckdtree.pyx", line 10, in init scipy.spatial._ckdtree
    from ._ckdtree import cKDTree, cKDTreeNode
  File "_ckdtree.pyx", line 10, in init scipy.spatial._ckdtree
    from ._ckdtree import cKDTree, cKDTreeNode
  File "_ckdtree.pyx", line 10, in init scipy.spatial._ckdtree
  File "/usr/local/lib/python3.8/dist-packages/scipy/sparse/__init__.py", line 283, in <module>
  File "/usr/local/lib/python3.8/dist-packages/scipy/sparse/__init__.py", line 283, in <module>
  File "/usr/local/lib/python3.8/dist-packages/scipy/sparse/__init__.py", line 283, in <module>
    from . import csgraph
  File "/usr/local/lib/python3.8/dist-packages/scipy/sparse/csgraph/__init__.py", line 185, in <module>
    from . import csgraph
  File "/usr/local/lib/python3.8/dist-packages/scipy/sparse/csgraph/__init__.py", line 185, in <module>
    from . import csgraph
  File "/usr/local/lib/python3.8/dist-packages/scipy/sparse/csgraph/__init__.py", line 185, in <module>
    from ._laplacian import laplacian
  File "/usr/local/lib/python3.8/dist-packages/scipy/sparse/csgraph/_laplacian.py", line 7, in <module>
    from ._laplacian import laplacian
  File "/usr/local/lib/python3.8/dist-packages/scipy/sparse/csgraph/_laplacian.py", line 7, in <module>
    from ._laplacian import laplacian
  File "/usr/local/lib/python3.8/dist-packages/scipy/sparse/csgraph/_laplacian.py", line 7, in <module>
    from scipy.sparse.linalg import LinearOperator
  File "/usr/local/lib/python3.8/dist-packages/scipy/sparse/linalg/__init__.py", line 120, in <module>
    from scipy.sparse.linalg import LinearOperator
  File "/usr/local/lib/python3.8/dist-packages/scipy/sparse/linalg/__init__.py", line 120, in <module>
    from scipy.sparse.linalg import LinearOperator
  File "/usr/local/lib/python3.8/dist-packages/scipy/sparse/linalg/__init__.py", line 120, in <module>
    from ._isolve import *
  File "/usr/local/lib/python3.8/dist-packages/scipy/sparse/linalg/_isolve/__init__.py", line 6, in <module>
    from ._isolve import *
  File "/usr/local/lib/python3.8/dist-packages/scipy/sparse/linalg/_isolve/__init__.py", line 6, in <module>
    from ._isolve import *
  File "/usr/local/lib/python3.8/dist-packages/scipy/sparse/linalg/_isolve/__init__.py", line 6, in <module>
    from .lgmres import lgmres
  File "/usr/local/lib/python3.8/dist-packages/scipy/sparse/linalg/_isolve/lgmres.py", line 7, in <module>
    from .lgmres import lgmres
  File "/usr/local/lib/python3.8/dist-packages/scipy/sparse/linalg/_isolve/lgmres.py", line 7, in <module>
    from .lgmres import lgmres
  File "/usr/local/lib/python3.8/dist-packages/scipy/sparse/linalg/_isolve/lgmres.py", line 7, in <module>
    from scipy.linalg import get_blas_funcs
  File "/usr/local/lib/python3.8/dist-packages/scipy/linalg/__init__.py", line 220, in <module>
    from scipy.linalg import get_blas_funcs
  File "/usr/local/lib/python3.8/dist-packages/scipy/linalg/__init__.py", line 220, in <module>
    from scipy.linalg import get_blas_funcs
  File "/usr/local/lib/python3.8/dist-packages/scipy/linalg/__init__.py", line 220, in <module>
    from . import (
ImportError: cannot import name 'misc' from partially initialized module 'scipy.linalg' (most likely due to a circular import) (/usr/local/lib/python3.8/dist-packages/scipy/linalg/__init__.py)

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/app/test_imports.py", line 9, in <module>
    from . import (
ImportError: cannot import name 'misc' from partially initialized module 'scipy.linalg' (most likely due to a circular import) (/usr/local/lib/python3.8/dist-packages/scipy/linalg/__init__.py)

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/app/test_imports.py", line 9, in <module>
    from . import (
ImportError: cannot import name 'misc' from partially initialized module 'scipy.linalg' (most likely due to a circular import) (/usr/local/lib/python3.8/dist-packages/scipy/linalg/__init__.py)

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/app/test_imports.py", line 9, in <module>
    from transformers import HfArgumentParser, TrainingArguments, Trainer
  File "<frozen importlib._bootstrap>", line 1039, in _handle_fromlist
  File "/usr/local/lib/python3.8/dist-packages/transformers/utils/import_utils.py", line 1376, in __getattr__
    from transformers import HfArgumentParser, TrainingArguments, Trainer
  File "<frozen importlib._bootstrap>", line 1039, in _handle_fromlist
  File "/usr/local/lib/python3.8/dist-packages/transformers/utils/import_utils.py", line 1376, in __getattr__
    from transformers import HfArgumentParser, TrainingArguments, Trainer
  File "<frozen importlib._bootstrap>", line 1039, in _handle_fromlist
  File "/usr/local/lib/python3.8/dist-packages/transformers/utils/import_utils.py", line 1376, in __getattr__
    module = self._get_module(self._class_to_module[name])
  File "/usr/local/lib/python3.8/dist-packages/transformers/utils/import_utils.py", line 1388, in _get_module
    module = self._get_module(self._class_to_module[name])
  File "/usr/local/lib/python3.8/dist-packages/transformers/utils/import_utils.py", line 1388, in _get_module
    module = self._get_module(self._class_to_module[name])
  File "/usr/local/lib/python3.8/dist-packages/transformers/utils/import_utils.py", line 1388, in _get_module
    raise RuntimeError(
RuntimeError: Failed to import transformers.trainer because of the following error (look up to see its traceback):
cannot import name 'misc' from partially initialized module 'scipy.linalg' (most likely due to a circular import) (/usr/local/lib/python3.8/dist-packages/scipy/linalg/__init__.py)
    raise RuntimeError(
RuntimeError: Failed to import transformers.trainer because of the following error (look up to see its traceback):
cannot import name 'misc' from partially initialized module 'scipy.linalg' (most likely due to a circular import) (/usr/local/lib/python3.8/dist-packages/scipy/linalg/__init__.py)
    raise RuntimeError(
RuntimeError: Failed to import transformers.trainer because of the following error (look up to see its traceback):
cannot import name 'misc' from partially initialized module 'scipy.linalg' (most likely due to a circular import) (/usr/local/lib/python3.8/dist-packages/scipy/linalg/__init__.py)
```
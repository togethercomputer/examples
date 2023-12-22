from dataclasses import dataclass, field
import os
from typing import Optional, List, Dict, Any

# The circular import problems arrise from the following relative imports using HuggingFace repos
# It appears to only happen on multinode jobs using apptainer where relative imports invocate
# the HuggingFace import_utils (e.g. transformers/utils/import_utils.py)
# So far this has been tested with accelerate and torchrun distributed launchers
from transformers import HfArgumentParser, TrainingArguments, Trainer
from transformers.trainer_utils import get_last_checkpoint
from transformers.deepspeed import is_deepspeed_zero3_enabled

from transformers import (
    HfArgumentParser,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    TrainerState,
    TrainerControl,
    AutoModelForCausalLM,
)

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from peft.tuners.lora import LoraLayer

from datasets import load_dataset
from transformers.testing_utils import CaptureLogger

from huggingface_hub import create_repo

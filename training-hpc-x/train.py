"""
This is a modified example of the following script, but for full model fine tuning of a CausalLM with FSDP and Flash Attention 2
https://github.com/huggingface/peft/blob/main/examples/conditional_generation/peft_lora_seq2seq_accelerate_fsdp.py
"""

import os

import torch
import argparse

from accelerate import Accelerator
from datasets import load_dataset, DatasetDict
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, default_data_collator, get_linear_schedule_with_warmup



def main(model_name_or_path, train_file, valid_file=None, valid_split=0.1, batch_size=8, 
         text_column="text", max_length=4096, lr=1e-3, num_epochs=1, use_flash_attn=False):
    accelerator = Accelerator()
    label_column = text_column # For CausalLM, the target is the original string

    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, use_flash_attention_2=use_flash_attn)

    # accelerator.print(model.print_trainable_parameters())

    dataset = DatasetDict()
    
    train_ext = os.path.splitext(train_file)[1]
    if valid_file:
        assert train_ext == os.path.splitext(valid_file)[1], "train and valid file must be of same type"
    
    if train_ext == 'jsonl':
        data_files = {"train": train_file}
        if valid_file:
            data_files["validation"] = valid_file
        dataset = load_dataset(
            "json",
            data_files=data_files
        )
    elif not train_ext:
        dataset = load_dataset(train_file)
        if valid_file:
            dataset["validation"] = load_dataset(valid_file)["train"]
    else:
        raise NotImplementedError(f"Do not support .{train_name_split[1]} extension")

    if "validation" not in dataset.keys():
        dataset = dataset["train"].train_test_split(test_size=valid_split)
        dataset["validation"] = dataset.pop("test")

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    # Add pad token if it does not exist
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer))

    def preprocess_function(examples):
        inputs = examples[text_column]
        targets = examples[label_column]
        model_inputs = tokenizer(
            inputs, max_length=max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        labels = tokenizer(targets, max_length=max_length, padding="max_length", truncation=True, return_tensors="pt")
        labels = labels["input_ids"]
        labels[labels == tokenizer.pad_token_id] = -100
        model_inputs["labels"] = labels
        return model_inputs

    with accelerator.main_process_first():
        processed_datasets = dataset.map(
            preprocess_function,
            batched=True,
            num_proc=1,
            remove_columns=dataset["train"].column_names,
            load_from_cache_file=False,
            desc="Running tokenizer on dataset",
        )

    train_dataset = processed_datasets["train"]
    eval_dataset = processed_datasets["validation"]

    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=default_data_collator, batch_size=batch_size, pin_memory=True
    )
    eval_dataloader = DataLoader(
        eval_dataset, collate_fn=default_data_collator, batch_size=batch_size, pin_memory=True
    )

    # When using FSDP, it is efficient and recommended to call prepare for the model before creating the optimizer
    model = accelerator.prepare(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=(len(train_dataloader) * num_epochs),
    )

    # When using FSDP, it is efficient and recommended to call prepare for the model before creating the optimizer
    train_dataloader, eval_dataloader, optimizer, lr_scheduler = accelerator.prepare(
        train_dataloader, eval_dataloader, optimizer, lr_scheduler
    )

    accelerator.print(model)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for step, batch in enumerate(tqdm(train_dataloader)):
            outputs = model(**batch)
            loss = outputs.loss
            total_loss += loss.detach().float()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        model.eval()
        eval_loss = 0
        eval_preds = []
        for step, batch in enumerate(tqdm(eval_dataloader)):
            with torch.no_grad():
                outputs = model(**batch)
            loss = outputs.loss
            eval_loss += loss.detach().float()
            preds = accelerator.gather_for_metrics(torch.argmax(outputs.logits, -1)).detach().cpu().numpy()
            eval_preds.extend(tokenizer.batch_decode(preds, skip_special_tokens=True))
        eval_epoch_loss = eval_loss / len(eval_dataloader)
        eval_ppl = torch.exp(eval_epoch_loss)
        train_epoch_loss = total_loss / len(train_dataloader)
        train_ppl = torch.exp(train_epoch_loss)
        accelerator.print(f"{epoch=}: {train_ppl=} {train_epoch_loss=} {eval_ppl=} {eval_epoch_loss=}")

        correct = 0
        total = 0
        for pred, true in zip(eval_preds, dataset["validation"][label_column]):
            if pred.strip() == true.strip():
                correct += 1
            total += 1
        accuracy = correct / total * 100
        accelerator.print(f"{accuracy=}")
        accelerator.print(f"{eval_preds[:10]=}")
        accelerator.print(f"{dataset['validation'][label_column][:10]=}")
        accelerator.wait_for_everyone()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test an LLM ckpt")
    parser.add_argument(
        "--model-name-or-path", type=str, default="meta-llama/Llama-2-7b-hf", help="model path"
    )
    parser.add_argument(
        "--train-file",
        type=str,
        default="togethercomputer/llama-instruct",
        help="Train dataset file (jsonl) or huggingface hub datset"
    )
    parser.add_argument(
        "--valid-file", type=str,
        default=None,
        help="Validation dataset file (jsonl) or huggingface hub datset. If None, uses a split of train."
    )
    parser.add_argument(
        "--valid-split", type=float,
        default=0.1,
        help="Percent of train to split in to validation if dedicated validation file not specified."
    )
    parser.add_argument(
        "--batch-size", type=int, default=8, help="Batch Size"
    )
    parser.add_argument(
        "--text-column", type=str, default="text", help="Column for the text in the datsets"
    )
    parser.add_argument(
        "--max-length", type=int, default=4096, help="Max sequence length."
    )
    parser.add_argument(
        "--learning-rate", type=float, default=1e-4, help="Learning Rate"
    )
    parser.add_argument(
        "--num-epochs", type=int, default=1, help="Number of epochs"
    )
    parser.add_argument(
        "--use-flash-attn", action="store_true", help="Use flash attention v2 (not supported by all models)"
    )

    args = parser.parse_args()

    main(args.model_name_or_path, args.train_file, args.valid_file, args.valid_split, args.batch_size, 
         args.text_column, args.max_length, args.learning_rate, args.num_epochs, args.use_flash_attn)
import os
import time
import json
import pickle
import random
import logging

import argparse
import together
import tabulate
import concurrent.futures
import numpy as np

from tqdm import tqdm
from typing import Any, Dict, List

from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import LocallyLinearEmbedding

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

from sklearn.metrics.pairwise import cosine_similarity

MSE_NAME = "Mean Squared Error"
COS_SIM_NAME = "Cosine Similarity"
DESCRIM_SCORE_NAME = "Descriminator Accuracy"

MAX_RETRIES=10

def load_jsonl_file(file_path, max_samples=-1):
    data = []
    with open(file_path, "r") as file:
        for i, line in enumerate(file):
            if max_samples > 0 and i >= max_samples:
                break
            data.append(json.loads(line))
    return data


def get_embeddings(texts: List[str], model: str, max_retries: int = MAX_RETRIES) -> List[List[float]]:
    i = 0
    retries = 0
    while retries < max_retries:
        try:
            outputs = together.Embeddings.create(input=texts, model=model)
            return [outputs.data[i].embedding for i in range(len(texts))]
        except:
            retries += 1
            time.sleep(10)

    raise Exception("Failed to retrieve embeddings")

def get_embeddings_batched(texts: List[str], model: str, batch_size: int, max_retries: int = MAX_RETRIES) -> List[List[float]]:
    if batch_size <= 0:
        batch_size = len(texts)
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        try:
            batch_embeddings = get_embeddings(batch, model, max_retries)
        except Exception as e:
            print(batch)
            raise Exception("Failed")
        embeddings.extend(batch_embeddings)
    return embeddings

def complete_prompt(
    prompt,
    model,
    max_tokens,
    temperature=0.8,
    top_k=60,
    top_p=0.6,
    repetition_penalty=1,
    stop=[],
    max_retries=MAX_RETRIES,
):
    retries = 0
    while retries < max_retries:
        try:
            output = together.Complete.create(
                prompt=prompt,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                stop=stop,
            )

            return output["output"]["choices"][0]["text"]
        except:
            retries += 1
            time.sleep(3)
    raise Exception("Failed to complete prompt")


def generate_responses_and_embs(
    gt_prompts, gt_responses, model, emb_model, stop_tokens, embedding_batch_size, max_threads=10
):
    print(f"Generating responses for {model}...")
    model_responses = []
    generate_model_responses = lambda gt_prompt, gt_response: complete_prompt(
        gt_prompt, model, max(len(gt_response), 10), stop=stop_tokens
    )

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_threads) as executor:
        futures = executor.map(generate_model_responses, gt_prompts, gt_responses)
        for response in tqdm(futures, total=len(gt_prompts)):
            model_responses.append(response)

    prompt_resp = [(prompt, resp) for prompt, resp in zip(gt_prompts, model_responses) if resp != ""]
    if len(prompt_resp) != len(gt_prompts):
        logging.warning(f"Failed to generate response for {len(gt_prompts) - len(prompt_resp)} prompts from {model}")
    filtered_resp = [resp for _,resp in prompt_resp]

    # Get base model embs
    print(f"Getting embeddings for {model} responses...")
    model_embs = get_embeddings_batched(filtered_resp, emb_model, embedding_batch_size)

    return {prompt: (resp, model_embs[i]) for i, (prompt, resp) in enumerate(prompt_resp)}


def generate_all_responses_and_embs(
    gt_prompts, gt_responses, ft_id, emb_model, stop_tokens, embedding_batch_size, max_threads=10, save_to_file=None
):
    # Get fine-tune model info
    job_info = together.Finetune.retrieve(fine_tune_id=ft_id)

    model_name = job_info["model_output_name"]
    base_model = job_info["model"]

    model_already_started = together.Models.instances()[model_name]
    if not model_already_started:
        # Start model
        print(f"Starting model {model_name}...")
        together.Models.start(model_name)

    # Get embeddings for gt_resonses
    print("Getting embeddings for ground truth responses...")
    gt_embs_all = get_embeddings_batched(gt_responses, emb_model, embedding_batch_size)
    gt_embs_dict = {prompt: (resp, emb) for prompt, resp, emb in zip(gt_prompts, gt_responses, gt_embs_all)}
    # Generate base model responses and embs
    print("Processing base model")
    base_model_resp_dict = generate_responses_and_embs(
        gt_prompts, gt_responses, base_model, emb_model, stop_tokens, embedding_batch_size, max_threads=max_threads
    )

    # Wait for ft model to be started
    print("Waiting for fine-tuned model to be started...")
    while not together.Models.instances()[model_name]:
        time.sleep(5)

    # Generate ft model responses
    print("Processing finetuned model")
    ft_model_resp_dict = generate_responses_and_embs(
        gt_prompts, gt_responses, model_name, emb_model, stop_tokens, embedding_batch_size, max_threads=max_threads
    )
    
    base_model_responses, base_model_embs = [], []
    ft_model_responses, ft_model_embs = [], []
    gt_embs = []
    for prompt, (resp, emb) in base_model_resp_dict.items():
        if prompt in ft_model_resp_dict:
            base_model_responses.append(resp)
            base_model_embs.append(emb)
            ft_model_responses.append(ft_model_resp_dict[prompt][0])
            ft_model_embs.append(ft_model_resp_dict[prompt][1])
            gt_embs.append(gt_embs_dict[prompt][1])
    if len(base_model_responses) != len(gt_prompts):
        logging.warning(f"Failed to generate response for {len(gt_prompts) - len(base_model_responses)} in total")
    
    ft_model_embs = np.array(ft_model_embs)
    base_model_embs = np.array(base_model_embs)
    gt_embs = np.array(gt_embs)

    if not model_already_started:
        # Stop the fine-tuned model
        print("Stopping model...")
        together.Models.stop(model_name)

    # Store responses and embs in a pickled dictionary and save it to the file
    if save_to_file is not None:
        data = {
            "gt_responses": gt_responses,
            "base_model_responses": base_model_responses,
            "ft_model_responses": ft_model_responses,
            "gt_embs": gt_embs,
            "base_model_embs": base_model_embs,
            "ft_model_embs": ft_model_embs,
        }
        with open(save_to_file, "wb") as f:
            pickle.dump(data, f)
    return (
        base_model_responses,
        ft_model_responses,
        gt_embs,
        base_model_embs,
        ft_model_embs,
    )


def get_metric_stats(metric, gt_embs, base_model_embs, ft_model_embs, table_title):
    gt_base_dist = metric(gt_embs, base_model_embs)
    gt_ft_dist = metric(gt_embs, ft_model_embs)
    base_ft_dist = metric(base_model_embs, ft_model_embs)

    # Compute mean and std of Euclidean pair-wise distance
    gt_base_mean = np.mean(gt_base_dist)
    gt_base_std = np.std(gt_base_dist)
    gt_ft_mean = np.mean(gt_ft_dist)
    gt_ft_std = np.std(gt_ft_dist)
    base_ft_mean = np.mean(base_ft_dist)
    base_ft_std = np.std(base_ft_dist)

    table = [
        [table_title, "GT - Base", "GT - FT", "Base - FT"],
        [
            "Mean",
            "{:10.4f}".format(gt_base_mean),
            "{:10.4f}".format(gt_ft_mean),
            "{:10.4f}".format(base_ft_mean),
        ],
        [
            "Std",
            "{:10.4f}".format(gt_base_std),
            "{:10.4f}".format(gt_ft_std),
            "{:10.4f}".format(base_ft_std),
        ],
    ]
    return table


def train_classifier(embs1, embs2, classifier="SVC"):
    # Combine gt_embs and ft_embs into a single dataset
    X = np.concatenate((embs1, embs2))
    # Create labels for the dataset (0 for gt_embs, 1 for ft_embs)
    y = np.concatenate((np.zeros(len(embs1)), np.ones(len(embs2))))
    # Create an instance of the Soft-Margin SVM classifier
    if classifier == "SVC":
        descrim = SVC(kernel="poly", C=1.0)
    elif classifier == "LogisticRegression":
        descrim = LogisticRegression()
    else:
        raise NotImplementedError(
            f"Invalid classifier {classifier}. Must be one of SVC or LogisticRegression"
        )
    # Train the classifier
    descrim.fit(X, y)
    # Get the accuracy of the classifier
    score = descrim.score(X, y)
    # Return the trained classifier and score
    return descrim, score


def train_classifiers(gt_embs, base_model_embs, ft_model_embs, classifier="SVC"):
    # Train 2 classifiers: one for gt_embs vs base_model_embs, one for gt_embs vs ft_model_embs
    print("Training classifiers...")
    if gt_embs.shape[0] < gt_embs.shape[1]:
        logging.warning("More features than samples, classifier may be singular or overfit!")

    gt_base_descrim, gt_base_score = train_classifier(gt_embs, base_model_embs)
    gt_ft_descrim, gt_ft_score = train_classifier(gt_embs, ft_model_embs)
    base_ft_descrim, base_ft_score = train_classifier(base_model_embs, ft_model_embs)
    return gt_base_descrim, gt_ft_descrim, base_ft_descrim, gt_base_score, gt_ft_score, base_ft_score


def decompose_embs(gt_embs, base_model_embs, ft_model_embs, vis_method="pca"):
    if vis_method == "PCA":
        # Perform PCA on gt_embs
        decomp = PCA(n_components=2)
    elif vis_method == "LLE":
        # Perform LLE on gt_embs
        decomp = LocallyLinearEmbedding(n_components=2)
    else:
        raise NotImplementedError(
            f"Invalid vis_method {vis_method}. Must be one of pca or lle"
        )
    gt_2d_result = decomp.fit_transform(gt_embs)
    base_model_2d_result = decomp.transform(base_model_embs)
    ft_model_2d_result = decomp.transform(ft_model_embs)

    return gt_2d_result, base_model_2d_result, ft_model_2d_result


def plot_all(
    gt_2d_result,
    base_model_2d_result,
    ft_model_2d_result,
    mse_table,
    cosine_table,
    score_table,
    image_file,
    vis_method,
):
    # Plot the PCA result and save to output_file

    plt.style.use("seaborn-v0_8")

    # Create subplots with 1 row and 2 columns
    fig = plt.figure(figsize=(12, 6))
    ax1 = plt.subplot2grid((3, 3), (0, 0), rowspan=3, colspan=2)
    ax2 = plt.subplot2grid((3, 3), (0, 2), rowspan=1)
    ax3 = plt.subplot2grid((3, 3), (1, 2), rowspan=1)
    ax4 = plt.subplot2grid((3, 3), (2, 2), rowspan=1)

    # Plot the PCA result
    ax1.scatter(gt_2d_result[:, 0], gt_2d_result[:, 1], label="Ground Truth", alpha=0.6)
    ax1.scatter(
        base_model_2d_result[:, 0],
        base_model_2d_result[:, 1],
        label="Base Model",
        alpha=0.6,
    )
    ax1.scatter(
        ft_model_2d_result[:, 0],
        ft_model_2d_result[:, 1],
        label="Fine-tuned Model",
        alpha=0.6,
    )
    ax1.legend()
    ax1.set_xlabel("Component 1")
    ax1.set_ylabel("Component 2")
    ax1.set_title(f"2D Visualization of Embeddings ({vis_method})")

    # Add euclidean distance table
    mse_col_labels = mse_table[0][1:]
    mse_row_labels = [row[0] for row in mse_table[1:]]
    mse_table = [row[1:] for row in mse_table[1:]]
    mse_table = ax2.table(
        cellText=mse_table,
        rowLabels=mse_row_labels,
        colLabels=mse_col_labels,
        loc="center",
    )
    mse_table.set_fontsize(16)
    mse_table.scale(1, 2)
    ax2.set_title(MSE_NAME)
    ax2.axis("off")

    # Add cosine similarity table
    cos_col_labels = cosine_table[0][1:]
    cos_row_labels = [row[0] for row in cosine_table[1:]]
    cosine_table = [row[1:] for row in cosine_table[1:]]
    cos_table = ax3.table(
        cellText=cosine_table,
        rowLabels=cos_row_labels,
        colLabels=cos_col_labels,
        loc="center",
    )
    cos_table.set_fontsize(16)
    cos_table.scale(1, 2)
    ax3.set_title(COS_SIM_NAME)
    ax3.axis("off")

    # Add score table
    score_col_labels = score_table[0][1:]
    score_row_labels = [row[0] for row in score_table[1:]]
    score_table = [row[1:] for row in score_table[1:]]
    score_table = ax4.table(
        cellText=score_table,
        rowLabels=score_row_labels,
        colLabels=score_col_labels,
        loc="center",
    )
    score_table.set_fontsize(16)
    score_table.scale(1, 2)
    ax4.set_title(DESCRIM_SCORE_NAME)
    ax4.axis("off")

    # Adjust the spacing between subplots
    plt.subplots_adjust(wspace=0.4)  # Decreased wspace value

    plt.savefig(image_file)


def main(
    examples,
    emb_model,
    image_file,
    ft_id=None,
    load_file=None,
    instruction_split=False,
    max_threads=1,
    save_file=None,
    vis_method="pca",
    classifier="SVC",
    stop_tokens=[],
    embedding_batch_size=-1,
):
    assert (
        ft_id is not None or load_file is not None
    ), "Must provide either ft_id or load_file"

    # Get ground truth prompts and responses
    print("Getting ground truth prompts and responses...")
    gt_texts = [example["text"] for example in examples]
    gt_prompts = []
    gt_responses = []
    for gt_text in gt_texts:
        if instruction_split:
            gt_prompt, gt_response = gt_text.split("[/INST]")
        else:
            split_point = random.randint(1, len(gt_text) - 1)
            gt_prompt, gt_response = gt_text[:split_point], gt_text[split_point:]
        gt_prompts.append(gt_prompt + "[/INST]")
        gt_responses.append(gt_response)

    if load_file is not None:
        # Load responses and embs from the file
        print("Loading responses and embs from the file...")
        with open(load_file, "rb") as f:
            data = pickle.load(f)
        gt_responses = data["gt_responses"]
        base_model_responses = data["base_model_responses"]
        ft_model_responses = data["ft_model_responses"]
        gt_embs = data["gt_embs"]
        base_model_embs = data["base_model_embs"]
        ft_model_embs = data["ft_model_embs"]
    else:
        results = generate_all_responses_and_embs(
            gt_prompts,
            gt_responses,
            ft_id,
            emb_model,
            stop_tokens,
            embedding_batch_size,
            max_threads=max_threads,
            save_to_file=save_file,
        )
        # Unpack the results
        (
            base_model_responses,
            ft_model_responses,
            gt_embs,
            base_model_embs,
            ft_model_embs,
        ) = results

    # Compute Euclidean pair-wise distance between 3 emb groups
    mse_table = get_metric_stats(
        lambda x, y: np.linalg.norm(x - y, axis=1),
        gt_embs,
        base_model_embs,
        ft_model_embs,
        MSE_NAME,
    )
    print("")
    print(tabulate.tabulate(mse_table, headers="firstrow"))
    print("")
    # Compute cosine similarity between 3 emb groups
    cosine_table = get_metric_stats(
        lambda x, y: cosine_similarity(
            x, y
        ).diagonal(),  # Diagonal to get only for matching pairs
        gt_embs,
        base_model_embs,
        ft_model_embs,
        COS_SIM_NAME,
    )
    print(tabulate.tabulate(cosine_table, headers="firstrow"))
    print("")
    # Get the classifiers and scores
    gt_base_descrim, gt_ft_descrim, base_ft_descrim, gt_base_score, gt_ft_score, base_ft_score = train_classifiers(
        gt_embs, base_model_embs, ft_model_embs, classifier=classifier
    )

    # Print the scores to stdout in table format
    score_table = [
        [DESCRIM_SCORE_NAME, "GT - Base", "GT - FT", "Base - FT"],
        ["Score", "{:10.4f}".format(gt_base_score), "{:10.4f}".format(gt_ft_score), "{:10.4f}".format(base_ft_score)],
    ]
    print("")
    print(tabulate.tabulate(score_table, headers="firstrow"))
    print("")

    # Perform PCA on gt_embs
    gt_2d_result, base_model_2d_result, ft_model_2d_result = decompose_embs(
        gt_embs, base_model_embs, ft_model_embs, vis_method=vis_method
    )

    plot_all(
        gt_2d_result,
        base_model_2d_result,
        ft_model_2d_result,
        mse_table,
        cosine_table,
        score_table,
        image_file,
        vis_method,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--examples-file", type=str, help="Path to test dataset")
    parser.add_argument(
        "--load-file",
        type=str,
        default=None,
        help="Path to a pickled dictionary containing responses and embs",
    )
    parser.add_argument("--ft-id", default=None, type=str, help="Fine-tune ID")
    parser.add_argument(
        "--instruction-split",
        action="store_true",
        help="Whether to split the text into prompt and response using the [/INST] token",
    )
    parser.add_argument(
        "--emb-model",
        type=str,
        default="togethercomputer/m2-bert-80M-2k-retrieval",
        help="Model name to use for embeddings",
    )
    parser.add_argument(
        "--image-file", type=str, default="embs-plot.png", help="Output image file name"
    )
    parser.add_argument(
        "--save-file",
        type=str,
        default=None,
        help="File to save pickled responses and embs to",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=-1,
        help="Maximum number of samples to use. Default is -1, which means use all samples",
    )
    parser.add_argument(
        "--max-threads",
        type=int,
        default=32,
        help="Number of threads to use for querying Together Complete",
    )
    parser.add_argument(
        "--vis-method",
        type=str,
        default="PCA",
        help="Visualization method to use.",
        choices=["PCA", "LLE"],
    )
    parser.add_argument(
        "--classifier",
        type=str,
        default="svc",
        help="Binary classifier to use.",
        choices=["SVC", "LogisticRegression"],
    )
    parser.add_argument(
        "--stop-tokens",
        nargs="+",
        help="List of stop tokens to use for prompt completion",
        default=[]
    )
    parser.add_argument(
        "--embedding-batch-size",
        type=int,
        default=-1,
        help="Batch size for generating embeddings. If <= 0, then no batching is used.",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=os.environ["TOGETHER_API_KEY"],
        help="Together API token",
    )

    args = parser.parse_args()

    together.api_key = args.token
    examples = load_jsonl_file(args.examples_file, max_samples=args.max_samples)
    client = together.Together()

    main(
        examples,
        args.emb_model,
        args.image_file,
        args.ft_id,
        args.load_file,
        args.instruction_split,
        args.max_threads,
        args.save_file,
        args.vis_method,
        args.classifier,
        args.stop_tokens,
        args.embedding_batch_size,
    )

import transformers
import datasets
import argparse
import wandb
import numpy as np
from datasets import load_metric
import os
from pytorch_metric_learning import losses
import sys
import logging
import torch
import torch.nn.functional as F


def mean_pooling(model_output, attention_mask):
    """
    This code is from Huggingface. NOT USED
    :url https://huggingface.co/sentence-transformers/all-mpnet-base-v2
    :param model_output:
    :param attention_mask:
    :return:
    """
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


class AdaptedTrainer(transformers.Trainer):
    def __init__(self, contrastive: bool = False, **kwds):
        super().__init__(**kwds)
        self.contrastive = contrastive

    def compute_loss(self, model, inputs, return_outputs=False):

        extra_loss_func = losses.SupConLoss() if self.contrastive else None

        if extra_loss_func is not None:
            loss, outputs = super().compute_loss(model, inputs, return_outputs=True)
            last_hidden_state = outputs,get("hidden_states")[-1]
            # last_hidden_state is of shape (batch_size, sequence_length, hidden_size).
            # we might like to use mean pooling, but I can't see how to do this generically
            # so we use CLS pooling

            # cls_representation is of shape (batch_size,hidden_size)
            cls_representation = last_hidden_state[:, 0, :]
            extra_loss = extra_loss_func(cls_representation,inputs,get("labels"))
            loss += extra_loss
            if return_outputs:
                return loss, outputs
            else:
                return loss

        elif extra_loss_func is None and return_outputs:
            loss, outputs = super().compute_loss(model, inputs, return_outputs=True)

            return loss, outputs
        else:
            return super().compute_loss(model, inputs, return_outputs=False)


def check_bool(s):
    s = s.lower()
    return s.startswith("t")


if __name__ == "__main__":
    wandb.init(project="intents", entity="cbrew81")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, default="sentence-transformers/all-MiniLM-L6-v2"
    )
    parser.add_argument("--dataset", type=str, default="banking77")
    parser.add_argument("--seed", type=int, default=12)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_epochs", type=int, default=40)
    parser.add_argument("--output_hidden_states", type=check_bool, default=False)
    parser.add_argument("--contrastive", type=check_bool, default=False)
    args = parser.parse_args()

    if args.contrastive and not args.output_hidden_states:
        logging.critical(
            "Contrastive learning requires that the model embeddings are returned"
        )
        sys.exit(-1)

    wandb.init(project="intents")
    model_name = args.model
    dataset_name = args.dataset
    dataset = datasets.load_dataset(dataset_name)
    labels = dataset["train"].features["label"].names
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    model = transformers.AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(labels),
        output_hidden_states=args.output_hidden_states,
    )

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)

    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    small_train_dataset = tokenized_datasets["train"].shuffle(seed=42)
    small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42)

    outdir = f"models/{model_name}/{dataset_name}/{wandb.run.name}/{wandb.run.id}"
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    training_args = transformers.TrainingArguments(
        report_to="wandb",
        num_train_epochs=args.num_epochs,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_steps=5000,
        seed=args.seed,
        per_device_eval_batch_size=args.batch_size,
        per_device_train_batch_size=args.batch_size,
        save_total_limit=5,
        warmup_ratio=0.05,
        output_dir=outdir,
    )

    metric = load_metric("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    trainer = AdaptedTrainer(
        model=model,
        args=training_args,
        train_dataset=small_train_dataset,
        eval_dataset=small_eval_dataset,
        compute_metrics=compute_metrics,
        contrastive=args.contrastive,
    )

    trainer.train()

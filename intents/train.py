import transformers
import datasets
import argparse
import wandb
import numpy as np
import os
from pytorch_metric_learning import losses
import sys
import logging
import torch
from sklearn.metrics import f1_score, accuracy_score
from accelerate import Accelerator

from intents.trainer import MultilabelTrainer


def mean_pooling(model_output, attention_mask):
    """
    This code is from Huggingface. NOT USED
    :url https://huggingface.co/sentence-transformers/all-mpnet-base-v2
    :param model_output:
    :param attention_mask:
    :return:
    """
    token_embeddings = model_output[0]
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )


class AdaptedTrainer(transformers.Trainer):
    def __init__(self, contrastive: bool = False, **kwds):
        super().__init__(**kwds)
        self.contrastive = contrastive

    def compute_loss(self, model, inputs, return_outputs=False):
        extra_loss_func = losses.SupConLoss() if self.contrastive else None
        if extra_loss_func is not None:
            loss, outputs = super().compute_loss(model, inputs, return_outputs=True)
            last_hidden_state = outputs.get("hidden_states")[-1]
            # last_hidden_state is of shape (batch_size, sequence_length, hidden_size).
            # we might like to use mean pooling, but I can't see how to do this generically
            # so we use CLS pooling

            # cls_representation is of shape (batch_size,hidden_size)
            cls_representation = last_hidden_state[:, 0, :]

            extra_loss = extra_loss_func(cls_representation, inputs.get("labels"))

            # measures to avoid GPU memory leak
            loss += extra_loss
            # don't need the temporaries.
            del last_hidden_state
            del cls_representation
            if return_outputs:
                return loss, outputs
                del extra_loss_func
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
    multi_label_tasks = {"ecthr_a", "ecthr_b", "eurlex", "unfair_tos"}

    wandb.init(project="intents", entity="cbrew81")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, default="sentence-transformers/all-MiniLM-L6-v2"
    )
    parser.add_argument("--dataset", type=str, default="banking77")
    parser.add_argument("--task", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
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
    dataset = (
        datasets.load_dataset(dataset_name)
        if args.task is None
        else datasets.load_dataset(dataset_name, args.task)
    )
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
        per_device_eval_batch_size=int(
            args.batch_size / max(torch.cuda.device_count(), 1)
        ),
        per_device_train_batch_size=int(
            args.batch_size / max(torch.cuda.device_count(), 1)
        ),
        save_total_limit=5,
        warmup_ratio=0.05,
        output_dir=outdir,
    )

    def compute_metrics_multiclass(p: transformers.EvalPrediction):
        logits = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.argmax(logits, axis=1)
        macro_f1 = f1_score(
            y_true=p.label_ids, y_pred=preds, average="macro", zero_division=0
        )
        micro_f1 = f1_score(
            y_true=p.label_ids, y_pred=preds, average="micro", zero_division=0
        )
        accuracy = accuracy_score(y_true=p.label_ids, y_pred=preds)
        return {"macro-f1": macro_f1, "micro-f1": micro_f1, "accuracy": accuracy_score}

    def compute_metrics_multiliabel(p: transformers.EvalPrediction):
        # Fix gold labels
        y_true = np.zeros(
            (p.label_ids.shape[0], p.label_ids.shape[1] + 1), dtype=np.int32
        )
        y_true[:, :-1] = p.label_ids
        y_true[:, -1] = (np.sum(p.label_ids, axis=1) == 0).astype("int32")
        # Fix predictions
        logits = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = (expit(logits) > 0.5).astype("int32")
        y_pred = np.zeros(
            (p.label_ids.shape[0], p.label_ids.shape[1] + 1), dtype=np.int32
        )
        y_pred[:, :-1] = preds
        y_pred[:, -1] = (np.sum(preds, axis=1) == 0).astype("int32")
        # Compute scores
        macro_f1 = f1_score(
            y_true=y_true, y_pred=y_pred, average="macro", zero_division=0
        )
        micro_f1 = f1_score(
            y_true=y_true, y_pred=y_pred, average="micro", zero_division=0
        )
        return {"macro-f1": macro_f1, "micro-f1": micro_f1}

    TrainerClass = (
        MultilabelTrainer if args.task in multi_label_tasks else AdaptedTrainer
    )
    trainer = TrainerClass(
        model=model,
        args=training_args,
        train_dataset=small_train_dataset,
        eval_dataset=small_eval_dataset,
        compute_metrics=compute_metrics_multiliabel
        if args.task in multi_label_tasks
        else compute_metrics_multiclass,
        contrastive=args.contrastive,
    )

    trainer.train()

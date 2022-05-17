import transformers
import datasets
import argparse
import wandb
import numpy as np
from datasets import load_metric
import os


class AdaptedTrainer(transformers.Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        if return_outputs:
            loss, outputs = super().compute_loss(model, inputs, return_outputs=True)
            return loss, outputs
        else:
            return super().compute_loss(model, inputs, return_outputs=False)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, default="sentence-transformers/all-MiniLM-L6-v2"
    )
    parser.add_argument("--dataset", type=str, default="banking77")
    parser.add_argument("--seed", type=int, default=12)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_epochs", type=int, default=40)
    args = parser.parse_args()

    wandb.init(project="intents")
    model_name = args.model
    dataset_name = args.dataset
    dataset = datasets.load_dataset(dataset_name)
    labels = dataset["train"].features["label"].names
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    model = transformers.AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=len(labels)
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
    )

    trainer.train()

import datetime
import os
import argparse
import sys
import pandas as pd
import datasets as datasets
import sentence_transformers as st
from sklearn.neighbors import KNeighborsClassifier
import sklearn.metrics as skm
import numpy as np
import transformers
import torch
import torch.nn.functional as F


def mean_pooling(model_out, attention_mask):
    token_embeddings = model_out[0]
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        nargs="+",
        default=[
            "nreimers/MiniLM-L6-H384-uncased",
            "sentence-transformers/all-MiniLM-L6-v2",
            "sentence-transformers/all-mpnet-base-v2",
            "microsoft/mpnet-base",
            "sentence-transformers/all-distilroberta-v1",
            "distilroberta-base",
            "sentence-transformers/all-MiniLM-L12-v2",
            "microsoft/MiniLM-L12-H384-uncased",
            "sentence-transformers/all-roberta-large-v1",
            "roberta-large",
        ],
        help="model to classify dataset with. Default is recommended transformer models and their base models.",
    )
    args = parser.parse_args()

    dataset_name = "banking77"
    data = datasets.load_dataset(dataset_name)
    # add a training item for each intent that is just the name of the
    # intent with underscores replaced by spaces.
    names = data["test"].features["label"].names
    for i, name in enumerate(names):
        data["train"] = data["train"].add_item(
            dict(text=" ".join(name.split("/")), label=i)
        )
    print(args.models)
    for model_name in args.models:
        DEVICE_CPU = -1

        print(model_name, dataset_name)

        targets = data["train"].features["label"].names

        y = data["test"]["label"]
        y_train = data["train"]["label"]
        if model_name.startswith("sentence-transformers"):
            model_st = st.SentenceTransformer(model_name)
            X_train = model_st.encode(data["train"]["text"])
            X_test = model_st.encode(data["test"]["text"])
        else:
            tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
            model = transformers.AutoModel.from_pretrained(model_name)
            X = dict()
            for partition in ("train", "test"):
                embeddings = []
                for x in data[partition]:
                    tokenized = tokenizer(x["text"], return_tensors="pt")
                    with torch.no_grad():
                        model_output = model(**tokenized)
                    embedding = mean_pooling(model_output, tokenized["attention_mask"])
                    embedding = F.normalize(embedding, p=2, dim=1)
                    embeddings.append(embedding.numpy())
                X[partition] = np.concatenate(embeddings, axis=0)
            X_train = X["train"]
            X_test = X["test"]

        date_str = datetime.datetime.now().isoformat()
        report_dir = os.path.join("reports", dataset_name, model_name, date_str)
        if not os.path.exists(report_dir):
            os.makedirs(report_dir)
        for k in range(1, 6):
            nn = KNeighborsClassifier(n_neighbors=k)
            nn.fit(X_train, y_train)
            y_hat = nn.predict(X_test)
            p_hat = nn.predict_proba(X_test)
            df = pd.DataFrame(
                skm.classification_report(
                    y, y_hat, output_dict=True, target_names=targets
                )
            ).T

            df.to_csv(os.path.join(report_dir, f"report_k{k}.csv"), encoding="utf8")
            predictions = pd.DataFrame(
                dict(
                    y=[names[i] for i in y],
                    yhat=[names[i] for i in y_hat],
                    model=model_name,
                )
            )
            for i in range(p_hat.shape[1]):
                predictions[names[i]] = p_hat[:, i]
            predictions.to_csv(
                os.path.join(report_dir, f"predictions_k{k}.csv"), encoding="utf8"
            )

import datasets as datasets
import sentence_transformers as st
from sklearn.neighbors import KNeighborsClassifier
import sklearn.metrics as skm
import argparse
import pandas as pd
import datetime
import os
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F

#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        nargs="+",
        default=[
            "sentence-transformers/all-MiniLM-L6-v2",
            "nreimers/MiniLM-L6-H384-uncased",
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

    print(args.models)
    for model_name in args.models:
        DEVICE_CPU = -1
        dataset_name = "banking77"
        print(model_name, dataset_name)
        data = datasets.load_dataset(dataset_name)

        targets = data["train"].features["label"].names

        if model_name.startswith('sentence-transformers'):
            model_st = st.SentenceTransformer(model_name)
            Xtrain = model_st.encode(data["train"]["text"])
            ytrain = data["train"]["label"]
            Xtest = model_st.encode(data["test"]["text"])
            y = data["test"]["label"]
        else:
            # Load model from HuggingFace Hub
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModel.from_pretrained(model_name)

            def preprocess(examples):
                return {""}

            encoded_data = data.map(preprocess)
            # Tokenize sentences

            # Compute token embeddings
            with torch.no_grad():
                model_output = model(**encoded_input)

            # Perform pooling
            sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

            # Normalize embeddings
            sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

        nn = KNeighborsClassifier(n_neighbors=1)
        nn.fit(Xtrain, ytrain)
        y_hat = nn.predict(Xtest)
        df = pd.DataFrame(
            skm.classification_report(y, y_hat, output_dict=True, target_names=targets)
        ).T
        date_str = datetime.datetime.now().isoformat()
        report_dir = os.path.join("reports",dataset_name, model_name, date_str)
        os.makedirs(report_dir)
        df.to_csv(os.path.join(report_dir, "report.csv"), encoding="utf8")
        predictions = pd.DataFrame(dict(y=np.array(y),
                                        yhat=y_hat, model=model_name))
        predictions.to_csv(os.path.join(report_dir, "predictions.csv"), encoding="utf8")

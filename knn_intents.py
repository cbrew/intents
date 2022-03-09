import datasets as datasets
import sentence_transformers as st
from sklearn.neighbors import KNeighborsClassifier
import sklearn.metrics as skm


if __name__ == "__main__":
    DEVICE_CPU = -1
    data = datasets.load_dataset("banking77")
    targets = data['train'].features['label'].names
    model_name = "sentence-transformers/all-mpnet-base-v2"
    model_st = st.SentenceTransformer(model_name)
    Xtrain = model_st.encode(data['train']['text'])
    ytrain = data['train']['label']
    nn = KNeighborsClassifier(n_neighbors=1)
    nn.fit(Xtrain, ytrain)
    yhat = nn.predict(model_st.encode(data['test']['text']))
    print(skm.classification_report(data['test']['label'], yhat,
                                    digits=5,
                                    target_names=targets))





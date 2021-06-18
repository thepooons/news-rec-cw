import numpy as np
import pandas as pd
import unicodedata
import re
from sklearn.cluster import KMeans
import yaml


def loadGloveModel(File):
    """loads a trained GloVe model

    Args:
        File (str): path to a trained GloVe model

    Returns:
        dict: word: vectors
    """
    print("Loading Glove Model")
    f = open(File, "r", encoding="utf-8")
    gloveModel = {}
    for line in f:
        splitLines = line.split()
        word = splitLines[0]
        wordEmbedding = np.array([float(value) for value in splitLines[1:]])
        gloveModel[word] = wordEmbedding
    print(len(gloveModel), " words loaded!")
    return gloveModel


if __name__ == "__main__":
    with open("config.yaml") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    GLOVE_VECTORS_PATH = config["glove_vectors_path"]
    RAW_DATA_PATH = config["raw_data_path"]
    VECTORED_DATA_PATH = config["clustered_vectorized_data_path"]

    # read the GloVe model and raw text data scraped from news websites
    glove_vectors = loadGloveModel(GLOVE_VECTORS_PATH)
    data = pd.read_csv(RAW_DATA_PATH)
    data = data.loc[:, ["heading", "content"]]

    headings_cleaned = []
    for heading in data.loc[:, "heading"]:
        heading = str(
            unicodedata.normalize("NFKD", heading)
            .encode("ascii", "ignore")
            .decode("ascii")
        ).lower()
        heading_cleaned = []
        tokens = [
            re.sub("""[!,*)@#+=~`%(&‘_\-:?.👏🏼“^”"'’\]\[]""", "", word.strip())
            for word in heading.split(" ")
        ]
        for token in tokens:
            if token in glove_vectors.keys():
                heading_cleaned.append(glove_vectors[token])
        headings_cleaned.append(np.mean(np.array(heading_cleaned), axis=0))

    contents_cleaned = []
    for content in data.loc[:, "content"]:
        content = str(
            unicodedata.normalize("NFKD", content)
            .encode("ascii", "ignore")
            .decode("ascii")
        ).lower()
        content_cleaned = []
        tokens = [
            re.sub("""[!,*)@#+=~`%(&‘_\-:?.👏🏼“^”"'’\]\[]""", "", word.strip())
            for word in content.split(" ")
        ]
        for token in tokens:
            if token in glove_vectors.keys():
                content_cleaned.append(glove_vectors[token])
        contents_cleaned.append(np.mean(np.array(content_cleaned), axis=0))

    heading_vectors = pd.DataFrame(
        data=np.array(headings_cleaned),
        columns=[f"heading_{i}" for i in range(50)],
    )
    content_vectors = pd.DataFrame(
        data=np.array(contents_cleaned),
        columns=[f"content_{i}" for i in range(50)],
    )

    vector_data = pd.concat(objs=[heading_vectors, content_vectors], axis=1)

    kmeanModel = KMeans(
        n_clusters=10,
        init="k-means++",
        n_init=10,
        max_iter=300,
        tol=0.0001,
        precompute_distances="deprecated",
        verbose=0,
        random_state=None,
        copy_x=True,
        n_jobs="deprecated",
        algorithm="auto",
    )
    kmeanModel.fit(vector_data)

    data = pd.read_csv(RAW_DATA_PATH)
    data.loc[:, "id"] = range(0, len(data))
    data.loc[:, "cluster_id"] = kmeanModel.labels_
    data.loc[:, [f"heading_{i}" for i in range(50)]] = vector_data.loc[
        :, [f"heading_{i}" for i in range(50)]
    ]
    data.loc[:, [f"content_{i}" for i in range(50)]] = vector_data.loc[
        :, [f"content_{i}" for i in range(50)]
    ]
    data.loc[:, "article_id"] = range(1, len(data) + 1)

    data.to_csv(config["clustered_vectorized_data_path"], index=False)

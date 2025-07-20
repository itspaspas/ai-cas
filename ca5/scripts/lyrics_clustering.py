# coding: utf-8
"""Lyrics Clustering Pipeline

This script loads the music lyrics dataset, preprocesses each lyric using
stemming or lemmatisation, produces sentence embeddings with the
all-MiniLM-L6-v2 SentenceTransformer and applies three clustering
algorithms (K-Means, DBSCAN and Agglomerative/Hierarchical).

It generates basic evaluation metrics and visualisations of the resulting
clusters.
"""

import argparse
import os
from pathlib import Path
import re
import string
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, homogeneity_score

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import pos_tag, word_tokenize
from sentence_transformers import SentenceTransformer

# Ensure required NLTK resources are available
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)

STOP_WORDS = set(stopwords.words('english'))
PUNCT_TABLE = str.maketrans('', '', string.punctuation)


def get_wordnet_pos(tag: str) -> str:
    """Map POS tag to first character lemmatize() accepts."""
    tag = tag[0].upper()
    return {
        'J': 'a',  # adjective
        'N': 'n',  # noun
        'V': 'v',  # verb
        'R': 'r',  # adverb
    }.get(tag, 'n')


def preprocess(text: str, *, method: str = 'lemma', ngram: int = 1) -> str:
    """Preprocess a single lyric.

    Parameters
    ----------
    text : str
        Raw lyric text.
    method : str
        Either ``'lemma'`` or ``'stem'``.
    ngram : int
        Size of n-grams to return.
    """
    tokens = [t.lower() for t in word_tokenize(text)]
    tokens = [t.translate(PUNCT_TABLE) for t in tokens]
    tokens = [t for t in tokens if t and t not in STOP_WORDS]

    if method == 'stem':
        stemmer = PorterStemmer()
        tokens = [stemmer.stem(t) for t in tokens]
    else:
        lemm = WordNetLemmatizer()
        pos_tags = pos_tag(tokens)
        tokens = [lemm.lemmatize(t, get_wordnet_pos(p)) for t, p in pos_tags]

    if ngram > 1:
        grams = [' '.join(tokens[i:i + ngram])
                 for i in range(len(tokens) - ngram + 1)]
    else:
        grams = tokens
    return ' '.join(grams)


def load_lyrics(csv_path: Path) -> pd.Series:
    """Load lyrics column from CSV."""
    df = pd.read_csv(csv_path)
    if 'Lyric' not in df.columns:
        raise ValueError('Expected column "Lyric" in dataset')
    df = df.dropna(subset=['Lyric'])
    return df['Lyric']


def embed_lyrics(lyrics: list[str], model_name: str) -> np.ndarray:
    model = SentenceTransformer(model_name)
    embeddings = model.encode(lyrics, show_progress_bar=True)
    return np.array(embeddings)


def run_kmeans(data: np.ndarray, k: int) -> tuple[np.ndarray, float]:
    km = KMeans(n_clusters=k, random_state=42)
    labels = km.fit_predict(data)
    score = silhouette_score(data, labels)
    return labels, score


def run_dbscan(data: np.ndarray, eps: float, min_samples: int) -> tuple[np.ndarray, float]:
    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(data)
    if len(set(labels)) > 1 and -1 not in set(labels):
        score = silhouette_score(data, labels)
    else:
        score = float('nan')
    return labels, score


def run_agglomerative(data: np.ndarray, k: int) -> tuple[np.ndarray, float]:
    agg = AgglomerativeClustering(n_clusters=k, linkage='ward')
    labels = agg.fit_predict(data)
    score = silhouette_score(data, labels)
    return labels, score


def elbow_plot(data: np.ndarray, k_range: range, out_path: Path) -> None:
    sse = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42)
        km.fit(data)
        sse.append(km.inertia_)
    plt.figure()
    plt.plot(list(k_range), sse, marker='o')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('SSE')
    plt.title('Elbow Method for K-Means')
    plt.savefig(out_path)
    plt.close()


def scatter_plot(data: np.ndarray, labels: np.ndarray, out_path: Path) -> None:
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(data)
    plt.figure()
    num_labels = len(set(labels))
    for lab in set(labels):
        idx = labels == lab
        plt.scatter(coords[idx, 0], coords[idx, 1], label=str(lab), s=10)
    if num_labels <= 10:
        plt.legend()
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def summarise_dataset(lyrics: pd.Series) -> dict:
    lengths = lyrics.str.split().str.len()
    return {
        'num_songs': len(lyrics),
        'min_length': lengths.min(),
        'max_length': lengths.max(),
        'avg_length': lengths.mean(),
    }


def main():
    parser = argparse.ArgumentParser(description='Lyrics Clustering Pipeline')
    parser.add_argument('--csv', type=Path, default=Path('musicLyrics.csv'),
                        help='Path to musicLyrics.csv')
    parser.add_argument('--model', default='all-MiniLM-L6-v2',
                        help='SentenceTransformer model name')
    parser.add_argument('--preprocess', choices=['stem', 'lemma'], default='lemma')
    parser.add_argument('--ngram', type=int, default=1)
    parser.add_argument('--out', type=Path, default=Path('outputs'))
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    lyrics = load_lyrics(args.csv)
    summary = summarise_dataset(lyrics)
    print(f"Loaded {summary['num_songs']} songs")
    print(f"Average length: {summary['avg_length']:.1f} words")

    processed = [preprocess(t, method=args.preprocess, ngram=args.ngram)
                 for t in tqdm(lyrics, desc='Preprocess')]

    embeddings = embed_lyrics(processed, args.model)

    # K-Means elbow
    elbow_plot(embeddings, range(2, 8), args.out / 'kmeans_elbow.png')

    k_labels, k_score = run_kmeans(embeddings, k=2)
    scatter_plot(embeddings, k_labels, args.out / 'kmeans_scatter.png')

    db_labels, db_score = run_dbscan(embeddings, eps=0.5, min_samples=5)
    if not np.isnan(db_score):
        scatter_plot(embeddings, db_labels, args.out / 'dbscan_scatter.png')

    ag_labels, ag_score = run_agglomerative(embeddings, k=3)
    scatter_plot(embeddings, ag_labels, args.out / 'agg_scatter.png')

    print('\nSilhouette Scores:')
    print(f'  K-Means (k=2): {k_score:.3f}')
    print(f'  DBSCAN: {db_score:.3f}')
    print(f'  Agglomerative (k=3): {ag_score:.3f}')

    # Example lyrics from first cluster of best model (k-means here)
    examples = defaultdict(list)
    for lyric, lab in zip(lyrics, k_labels):
        if len(examples[lab]) < 2:
            examples[lab].append(lyric[:120] + '...')
        if all(len(v) >= 2 for v in examples.values()):
            break

    print('\nCluster Examples (K-Means):')
    for lab, exs in examples.items():
        print(f'Cluster {lab}')
        for ex in exs:
            print(f'  - {ex}')


if __name__ == '__main__':
    main()

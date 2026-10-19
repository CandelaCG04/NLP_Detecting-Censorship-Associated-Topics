# Topic model

BERTopic fitted by `scripts/bertopic1.py` on `datasets/clean/topic_train.csv` (5,240 books), without ban labels. All values below are the script defaults.

## Embeddings

`sentence-transformers/all-mpnet-base-v2`: a general-purpose English sentence encoder with 768 dimensions, one of the strongest in the sentence-transformers benchmarks. Its 384-token limit covers nearly every text (median about 125 words), so summaries are not truncated. Embeddings are cached in `embeddings_{train,test}.npy`. Delete these files after changing the texts or the model.

## Seed

`--seed 42` is passed to UMAP as `random_state`. UMAP is the only random part of the pipeline: HDBSCAN and c-TF-IDF are deterministic. Running again with the same seed reproduces identical outputs.

## Pipeline

| Stage | Component | Parameters |
|-------|-----------|------------|
| Dimensionality reduction | UMAP | `n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine"` |
| Clustering | HDBSCAN | `min_cluster_size=25` (`--min-topic-size`), `metric="euclidean"`, `cluster_selection_method="eom"`, `prediction_data=True` |
| Topic words | CountVectorizer + c-TF-IDF | `stop_words="english", ngram_range=(1, 2), min_df=5` |

`min_cluster_size` is the smallest group of books that counts as a topic. Raising it gives fewer, broader topics, and lowering it gives more, narrower ones.

## Fit and transform

- **Train:** `fit_transform` on the training texts only. Outliers (topic -1)
  are reassigned to their closest topic by embedding (`reduce_outliers`, `strategy="embeddings"`), and the topic words are then recomputed (`update_topics`).
- **Test:** only `transform`, so the model never sees these books during fitting. Test outliers are mapped to the topic with the highest cosine similarity.
- **Topic shares:** `approximate_distribution(window=4, stride=1)` for both splits, with each row normalised to sum to 1. A book where no topic matched gets 1.0 on its assigned topic. Output: `train_topic_dist.csv` and `test_topic_dist.csv` (`Book ID, Banned Status, topic_0 … topic_23`).

## Results

24 topics (see `topic_info.csv`). Stability across refits with UMAP seeds 0, 1 and 2 (`--stability-seeds 0 1 2`):

| Seeds | Topics | ARI | Top-10-word Jaccard |
|-------|--------|-----|---------------------|
| 0 vs 1 | 25 / 24 | 0.807 | 0.775 |
| 0 vs 2 | 25 / 22 | 0.761 | 0.700 |
| 1 vs 2 | 24 / 22 | 0.804 | 0.727 |
| **mean** | | **0.791** | **0.734** |

## Reload

```python
from bertopic import BERTopic
model = BERTopic.load("datasets/topics/model")
```

`model/` and the embedding caches are git-ignored. Run
`python scripts/bertopic1.py` to recreate them.

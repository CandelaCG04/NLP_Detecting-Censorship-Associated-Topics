# NLP_Detecting-Censorship-Associated-Topics

## Datasets

The PEN America ban lists (`datasets/raw/banned/`) and the CMU Book Summary
Dataset (`datasets/raw/books/booksummaries.txt`) are already in the repo.
The Goodreads files are too large to commit and must be downloaded into
`datasets/raw/books/` before running `books_preprocess.py`.

| File | Source | Notes |
|------|--------|-------|
| `goodreads_books.json` | [goodreads_books.json.gz](https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/goodreads_books.json.gz) | ~2 GB compressed, ~9 GB extracted. **Must be decompressed** (the script reads the plain `.json`). |
| `goodreads_book_authors.json.gz` | [goodreads_book_authors.json.gz](https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/goodreads_book_authors.json.gz) | Keep compressed. |
| `goodreads_book_genres_initial.json.gz` | [goodreads_book_genres_initial.json.gz](https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/goodreads_book_genres_initial.json.gz) | Keep compressed. Optional: without it, Goodreads genres are left empty. |

All three come from the [UCSD Book Graph](https://mengtingwan.github.io/data/goodreads.html)
(Wan & McAuley, RecSys 2018). To download them:

```bash
cd datasets/raw/books
BASE=https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads
curl -LO $BASE/goodreads_books.json.gz
curl -LO $BASE/goodreads_book_authors.json.gz
curl -LO $BASE/goodreads_book_genres_initial.json.gz
gunzip goodreads_books.json.gz
```

Already included, for reference:

- **PEN America Index of School Book Bans** (2021–2025): <https://pen.org/book-bans/>
- **CMU Book Summary Dataset**: <https://www.cs.cmu.edu/~dbamman/booksummaries.html>

The first run of `books_preprocess.py` parses the full Goodreads file and caches
the result in `datasets/interim/goodreads_clean.pkl`; delete it to force a re-parse.

## Table columns, raw data

| Author | Title | Type of Ban | Secondary Author(s) | Date of Challenge/Removal | Ban Status |

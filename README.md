# NLP_Detecting-Censorship-Associated-Topics

Builds a dataset for studying which topics are associated with book censorship.
Books banned in US schools (PEN America) are paired with their plot summaries
(CMU Book Summary Dataset and Goodreads). Each banned book is matched to a
non-banned control book with similar genres and publication date.

## Setup

Requires Python 3.12+ (developed on 3.12.2).

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

All scripts are run from the repository root, and their default paths are
relative to it.

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

## Repository layout

```
datasets/
  raw/banned/      PEN America yearly CSVs (BannedBooks2021-2022.csv ... 2024-2025.csv)
  raw/books/       booksummaries.txt (CMU) + Goodreads files (downloaded, git-ignored)
  interim/         goodreads_clean.pkl cache (git-ignored)
  clean/           pipeline outputs
csv_preprocess.py    step 1: clean the PEN America ban lists
books_preprocess.py  step 2: clean and merge CMU + Goodreads summaries
merge.py             step 3: match banned books to summaries and build the control group
```

## Pipeline

Run the three steps in order. With no arguments, each script uses the default
paths shown. Use `-h` on any script to see its options.

```bash
python csv_preprocess.py     # -> datasets/clean/bannedT_titles_clean.csv
python books_preprocess.py   # -> datasets/clean/books_summaries_merged.csv
python merge.py              # -> datasets/clean/final_book_dataset.csv
```

Steps 1 and 2 are independent. Step 3 needs both of their outputs.

### 1. `csv_preprocess.py`: clean the ban lists

**Input:** every CSV in `datasets/raw/banned/` (or the files/folders passed as arguments).
**Output:** `datasets/clean/bannedT_titles_clean.csv` (override with `-o`), one row per unique book.

- Harmonises column names, which change between the yearly files.
- Normalises text: Unicode, whitespace, quotes, and ALL-CAPS or all-lower-case values.
- Cleans titles: removes bracketed series names ("Sloppy Firsts (Jessica Darling Series)" → "Sloppy Firsts") and fixes inverted articles ("Hobbit, The" → "The Hobbit").
- Flips authors from "Last, First" to "First Last" so they match CMU/Goodreads. A few multi-author entries can't be flipped and are kept as they are.
- Parses challenge dates (e.g. `Apr-22`, `June 2025`) to `YYYY-MM`.
- De-duplicates by author and title. Title variants by the same author are merged when their similarity is at least `--fuzzy-threshold` (default `0.92`; `1` disables this). Ban types and statuses from merged rows are joined with `|`, and the earliest challenge date is kept.

| Author | Title | Type of Ban | Secondary Author(s) | Date of Challenge/Removal | Ban Status |
|--------|-------|-------------|---------------------|---------------------------|------------|

### 2. `books_preprocess.py`: merge the summary datasets

**Inputs:** `datasets/raw/books/booksummaries.txt` and the three Goodreads files.
**Output:** `datasets/clean/books_summaries_merged.csv`.

- Drops books missing an author, title or summary.
- Matches CMU and Goodreads books by fuzzy author and title, comparing each title with and without its subtitle. A pair counts as the same book when the title score is at least `--title-threshold` (default 90) and the author score is at least `--author-threshold` (default 85). Each book is matched at most once; ties go to the more-rated Goodreads edition.
- Publication Date and Genres come from CMU when available, otherwise from Goodreads. Goodreads dates are those of the most-rated edition, not the original work. Dates are `YYYY`, `YYYY-MM` or `YYYY-MM-DD`, and genres are joined with `; `.
- Books found in only one dataset are kept, with the other summary left empty. Use `--matched-only` to keep only books found in both.
- The first run parses the full Goodreads file and caches the result in `datasets/interim/goodreads_clean.pkl`. Delete the cache (or point `--cache` elsewhere) to force a re-parse.

| Author | Title | Publication Date | Genres | CMU Summary | Goodreads Summary |
|--------|-------|------------------|--------|-------------|-------------------|

### 3. `merge.py`: banned books and control group

**Inputs:** the outputs of steps 1 and 2.
**Output:** `datasets/clean/final_book_dataset.csv`, sorted by author and title.

1. **Find banned books' summaries.** Each banned book is fuzzy-matched to the summaries data, using the same matching and thresholds as step 2. The matched book gets `Banned Status = True`. Banned books with no summary are dropped.
2. **Build the control group.** Each banned book is paired with exactly one non-banned book, and no control book is used twice:
   - The control is the non-banned book sharing the **most genres** with the banned book. Ties go to the **closest publication year**.
   - Matching is greedy, so order matters: banned books with the most genres pick first.
   - A banned book missing genres or a date is only matched against controls missing the same fields:

     | Banned book has | Control chosen among books with | Chosen by |
     |-----------------|---------------------------------|-----------|
     | genres and date | genres and date | most shared genres, then closest year |
     | no genres | no genres | closest year |
     | no date | no date | most shared genres |
     | neither | neither | random |

   - Remaining ties are broken randomly, and `--seed` (default `42`) makes the result reproducible.
   - Every fallback match (any row except the first) is printed to the terminal, along with any banned book left without a control. A summary at the end gives the count per match type and the number of matches that share no genre.

Options: `--banned`, `--summaries`, `-o/--output`, `--seed`, `--title-threshold`, `--author-threshold`.

The run takes a few minutes, mostly for the control-group matching against about 1.2M summaries.

| Author | Title | Publication Date | Genres | CMU Summary | Goodreads Summary | Banned Status |
|--------|-------|------------------|--------|-------------|-------------------|---------------|

With the current data, 3,455 of the 8,102 banned books have a summary. The
final dataset has 6,910 rows: 3,455 banned and 3,455 controls.

import csv
from tqdm import tqdm
import pandas as pd

def preprocess_CMU_data():
    CMU_data = []

    with open("datasets_raw/books/booksummaries.txt", 'r', encoding='utf-8') as f:
        reader = csv.reader(f, dialect='excel-tab')
        for row in tqdm(reader):
            CMU_data.append(row)

    return CMU_data

def see_CMU_sample(CMU_data):
    columns = ["Wikipedia ID", "Freebase ID", "Book Title", "Author", "Publication Date", "Genres", "Summary"]

    df = pd.DataFrame(CMU_data, columns=columns)

    selected_columns = ["Book Title", "Author", "Summary"]

    df_Selected = pd.DataFrame(df, columns=selected_columns)

    print(df.head())
    print(df_Selected.head())

preprocess_CMU_data()
see_CMU_sample(preprocess_CMU_data())
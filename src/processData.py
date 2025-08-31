import os
import pandas as pd
import secrets


SCENE_DETAILS_FOLDER = r"Z:\Projects\VideoSplitter\data"
chunksize = 10000

TSV_FILES = []

def refresh_TSV_list():
    global TSV_FILES
    TSV_FILES = []
    TSV_FILES = [os.path.join(SCENE_DETAILS_FOLDER,x) for x in os.listdir(SCENE_DETAILS_FOLDER) if x.endswith(".tsv")]


def generate_unique_filename(existing_names=None, length=32, ext=".tsv"):
    if existing_names is None:
        existing_names = set()
    while True:
        name = secrets.token_hex(length // 2)  # 32 hex chars
        filename = name + ext
        if filename not in existing_names:
            return filename

def read_complete_df():
    global TSV_FILES
    refresh_TSV_list()
    if len(TSV_FILES) == 0:
        print("No TSV files found.")
        return pd.DataFrame(columns=['name', 'scores'])
    dfs = [pd.read_csv(file, sep="\t") for file in TSV_FILES]
    combined_df = pd.concat(dfs, ignore_index=True)
    print(f"Loaded {len(combined_df)} rows .....")
    return combined_df

def save_new_temp_df(DF):
    unique_name = generate_unique_filename(set(os.path.basename(x) for x in TSV_FILES))
    DF.to_csv(os.path.join(SCENE_DETAILS_FOLDER, unique_name), sep="\t", index=False)


def deduplicate_df(df):
    before = len(df)
    df_dedup = df.drop_duplicates()
    after = len(df_dedup)
    print(f"Removed {before - after} duplicate rows.")
    return df_dedup



def split_and_save_df_chunks(chunk_size=chunksize):
    global TSV_FILES
    df = read_complete_df()
    df = deduplicate_df(df)
    num_chunks = (len(df) + chunk_size - 1) // chunk_size
    for i in range(num_chunks):
        chunk = df.iloc[i*chunk_size : (i+1)*chunk_size]
        filename = f"chunk_{i+ 1}.tsv"
        chunk.to_csv(os.path.join(SCENE_DETAILS_FOLDER, filename), sep="\t", index=False)
    for file in TSV_FILES:
        if len(os.path.basename(file)) == 36:
            os.remove(file)

refresh_TSV_list()
import os, re
import pandas as pd
import secrets


SCENE_DETAILS_FOLDER = r"Z:\Projects\VideoSplitter\data"
chunksize = 5000
TAG_REPLACEMENTS_DATA = r"data\tagReplacements.txt"
TAG_REPLACEMENTS = {}

print("\nLoading tag replacements ...")
with open(TAG_REPLACEMENTS_DATA, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if line and not line.startswith('#'):
            parts = line.split(':')
            if len(parts) == 2:
                key, value = parts
                TAG_REPLACEMENTS[key.strip()] = value.strip()

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
    print(f" - Removed {before - after} duplicate rows.")
    return df_dedup

def apply_tag_replacements(df, replacements):
    if not replacements:
        return df
    obj_cols = df.select_dtypes(include=['object']).columns
    if len(obj_cols) == 0:
        return df

    # Single compiled pattern for all keys, with word boundaries for exact tag hits
    pattern = re.compile(r'\b(' + '|'.join(map(re.escape, replacements.keys())) + r')\b')

    df[obj_cols] = df[obj_cols].apply(
        lambda s: s.str.replace(pattern, lambda m: replacements[m.group(0)], regex=True)
    )
    return df

def split_and_save_df_chunks(chunk_size=chunksize):
    global TSV_FILES
    df = read_complete_df()

    # Apply all tag replacements efficiently before deduping/chunking
    print("\n - Applying tag replacements ...")
    df = apply_tag_replacements(df, TAG_REPLACEMENTS)

    df = deduplicate_df(df)
    num_chunks = (len(df) + chunk_size - 1) // chunk_size
    print(f" - Splitting into {num_chunks} chunks of up to {chunk_size} rows each ...")
    for i in range(num_chunks):
        chunk = df.iloc[i*chunk_size : (i+1)*chunk_size]
        filename = f"chunk_{i+ 1}.tsv"
        chunk.to_csv(os.path.join(SCENE_DETAILS_FOLDER, filename), sep="\t", index=False)
    
    for file in TSV_FILES:
        if len(os.path.basename(file)) == 36:
            os.remove(file)
        elif "chunk" not in os.path.basename(file):
            print(f"Skipping removal of {file} with length {len(os.path.basename(file))}")
    print(" - Finished processing and saving chunks..\n ")

     

refresh_TSV_list()

# value_count = {}

# for key in TAG_REPLACEMENTS:
#     value = TAG_REPLACEMENTS[key]
#     value_count[value] = value_count.get(value, 0) + 1

# print("Tag replacements summary:")
# for value, count in value_count.items():
#     print(f"{value}:{count}")
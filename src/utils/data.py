import pandas as pd
from src.settings import *

def read_and_prepare_metadata():
    df = pd.read_csv(DATASET_ROOT + "/" + METADATA_FILENAME)

    df['location'] = df['location'].apply(lambda s: tuple(map(float, s.strip('()').split(', '))))
    df[CLASS_IDX_COLUMN_NAME] = df['class'].map(DEFAULT_CLASS_MAPPING)

    return df

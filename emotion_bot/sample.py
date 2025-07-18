import pandas as pd
from datasets import load_dataset

# Load full GoEmotions dataset
ds = load_dataset("go_emotions", split="train")


df = pd.DataFrame(ds)
print(df.head())       
print(df['labels'])
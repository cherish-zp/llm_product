from datasets import load_dataset
import random
import pandas as pd
import datasets
from IPython.display import display, HTML



# 下载数据的默认路径  ~/.cache/huggingface/datasets/yelp_review_full
dataset = load_dataset("yelp_review_full")

print(dataset)


def show_random_elements(dataset, num_examples=10):
    assert num_examples <= len(dataset), "Can't pick more elements than there are in the dataset."
    picks = []
    for _ in range(num_examples):
        pick = random.randint(0, len(dataset) - 1)
        while pick in picks:
            pick = random.randint(0, len(dataset) - 1)
        picks.append(pick)

    df = pd.DataFrame(dataset[picks])
    for column, typ in dataset.features.items():
        if isinstance(typ, datasets.ClassLabel):
            df[column] = df[column].transform(lambda i: typ.names[i])
    display(HTML(df.to_html()))
    print(df.to_html())
    #print(HTML(df.to_html()))

show_random_elements(dataset["train"])

from transformers import AutoTokenizer, AutoModel

tokenizer =AutoTokenizer.from_pretrained("bert-base-cased")
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)


tokenized_datasets = dataset.map(tokenize_function, batched=True)

show_random_elements(tokenized_datasets["train"], num_examples=1)

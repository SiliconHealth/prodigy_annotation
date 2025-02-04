from base.dataset.dataset import ListDataset
from base.dataset.huggingface_dataset import HuggingfaceDataset
from base.training.example import Instance
from ner.annotator.prodigy_ner_annotator import ProdigyNerAnnotator
from ner.bases.dataset_serializer import NerDatasetSerializer
import pandas as pd 

input_file = '../data/newest-1000-1.txt'
dataset = 'pii_0'


# Open the file in read mode
with open(input_file, 'r') as file:
    # Read all lines into a list; each element is a line from the file
    lines = file.readlines()

# Optionally, strip newline characters from each line
data = [line.strip() for line in lines]

annotator = ProdigyNerAnnotator(
    dataset=ListDataset(
        [Instance(input=d, label='') for d in data]
    ),
    dataset_name=dataset,
    config={
        "labels": ["tel", 'per', 'pla', 'sex', 'age', 'dat', 'nat', 'id'],
        "custom_theme": {
            "labels": {
                "tel": "lightgreen",
                "per": "lightblue",
                "pla": "lightgrey",
                "sex": "yellow",
                "age": "brown",
                "dat": "pink",
                "nat": "purple",
                "id": "orange",
            }
        }
    },
    buffer=NerDatasetSerializer(HuggingfaceDataset())
)

annotator.reset()
annotator.start()
from base.dataset.dataset import ListDataset
from base.dataset.huggingface_dataset import HuggingfaceDataset
from base.training.example import Instance
from ner.annotator.prodigy_ner_annotator import ProdigyNerAnnotator
from ner.bases.dataset_serializer import NerDatasetSerializer
import pandas as pd 

input_file = './data/pii_0_annotation.txt'
dataset = 'pii_0'


data = []
df = pd.read_csv(input_file, sep=',', header=None, encoding='utf-8')
data = df[0].tolist()
data = [str(d) for d in data if d]

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
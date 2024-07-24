from base.dataset.dataset import ListDataset
from base.dataset.huggingface_dataset import HuggingfaceDataset
from base.training.example import Instance
from ner.annotator.prodigy_ner_annotator import ProdigyNerAnnotator
from ner.bases.dataset_serializer import NerDatasetSerializer
import pandas as pd 


data = []
df = pd.read_csv('../data_for_annotation/med_0_annotation.txt', sep=',', header=None, encoding='utf-8')
data = df[0].tolist()

annotator = ProdigyNerAnnotator(
    dataset=ListDataset(
        [Instance(input=d, label='') for d in data]
    ),
    dataset_name="med_0",
    config={
        "labels": ["drug", 'str', 'route', 'dose', 'time'],
        "custom_theme": {
            "labels": {
                "drug": "lightgreen",
                "str": "lightblue",
                "route": "lightgrey",
                "dose": "yellow",
                "time": "lightbrown",
            }
        }
    },
    buffer=NerDatasetSerializer(HuggingfaceDataset())
)

annotator.reset()
annotator.start()
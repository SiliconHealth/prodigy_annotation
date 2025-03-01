from datetime import date
from base.dataset.dataset import ListDataset
from base.training.example import Instance
from classifier.annotator.prodigy_text_classifier import ProdigyTextClassificationAnnotator
import pandas as pd
import numpy as np


data = pd.read_csv('../data/med_error_text_classifier_dataset/med_classification_tanupat.csv', encoding='utf-8')
data = data.dropna(subset=['risk_all'])
data = data['risk_all'].tolist()

annotator = ProdigyTextClassificationAnnotator(
    dataset=ListDataset(
        [Instance(input=d, label=[]) for d in data]
    ),
    dataset_name="med_error_1",
    config={'options':[
    {"id": "wrong_med", "text": "Wrong Medication"},
    {"id": "wronge_patient", "text": "Wrong Patient"},
    {"id": "wrong_dose", "text": "Wrong Dose"},
    {"id": "wrong_strength", "text": "Wrong Strength"}, 
    {"id": "wrong_route", "text": "Wrong Route"},
    {"id": "wrong_time", "text": "Wrong Time"},
    {"id": "wrong_amount", "text": "Wrong Amount"},
    {"id": "omit", "text": "Omitted Medication"},
    {"id": "unneccessary", "text": "Unnecessary Medication"},
    {"id": "ddi", "text": "Drug-Drug Interaction"},
    {"id": "allergy", "text": "Allergic Reaction"},
    {"id": "ci", "text": "Contraindication"},
    {"id": "adr", "text": "Adverse Drug Reaction"},
    {"id": "none", "text": "None of the above"}, 
  ], 'choice_style': 'single'
    },
    buffer=ListDataset([]),
)

annotator.reset()
annotator.start()
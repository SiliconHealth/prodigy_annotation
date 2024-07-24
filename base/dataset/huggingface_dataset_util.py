import csv

from datasets import Dataset


def csv_to_hf(file):
    def gen():
        reader = csv.reader(file)
        cols = []
        for i, row in enumerate(reader):
            if i == 0:
                cols = [row[0][1:]] + row[1:]
            else:
                yield {cols[i]:v for i,v in enumerate(row)}

    return Dataset.from_generator(gen)
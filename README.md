# SiliconHealth NLP

This repository includes prodigy annotation for NER. 

## Usage
install prodigy before use. 

In the `main.py` file, you should change `input_file` and `dataset`.

## RUN 
```
$ python -m main.py
```

## Export data
dataset is the same as `dataset`.
file_name is a jsonl.
```
$ prodigy db-out <dataset> > <file_name>
```
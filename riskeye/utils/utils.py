import csv
import gzip
import pandas as pd

def get_header_length_csv(fn):
    return pd.read_csv(fn, delim_whitespace=True, nrows=0).shape[1]
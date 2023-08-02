import os.path as op
import pandas as pd

def get_all_subject_ids():
    return [f'{e:02d}' for e in range(1, 65)]

def get_all_subject(bids_folder='/data/ds-riskeye'):
    return [Subject(e, bids_folder=bids_folder) for e in get_all_subject_ids()]

def get_all_behavior(include_no_responses=False, bids_folder='/data/ds-riskeye'):
    return pd.concat([e.get_behavior(include_no_responses=include_no_responses) for e in get_all_subject(bids_folder=bids_folder)])

class Subject(object):


    def __init__(self, subject_id, bids_folder='/data/ds-riskeye'):

        if type(subject_id) is int:
            subject_id = f'{subject_id:02d}'

        self.subject_id = subject_id
        self.bids_folder = bids_folder


    def get_behavior(self, include_no_responses=False):

        df = pd.read_csv(op.join(self.bids_folder, f'sub-{self.subject_id}', 'func',
                                   f'sub-{self.subject_id}_events.tsv'), sep='\t',
                                   index_col=['subject', 'run', 'exptype', 'block', 'trial_nr'])

        if not include_no_responses:
            df = df[~df['chose_risky'].isnull()]

        return df
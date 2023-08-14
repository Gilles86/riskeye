import os
import os.path as op
from tqdm import tqdm
import argparse
from riskeye.utils.data import get_all_subjects
import pandas as pd


def main(source, summarize, root_folder='/data/ds-riskeye'):

    subjects = get_all_subjects(root_folder)

    target_dir = op.join(root_folder, 'derivatives', 'pupil')
    if not op.exists(target_dir):
        os.makedirs(target_dir)

    df = []

    for subject in tqdm(subjects):
        try:
            df.append(subject.get_trialwise_saccade_info(source=source, summarize_trials=summarize))
        except Exception as e:
            print(e)
            print(f'Could not load data for subject {subject.subject_id}')

    df = pd.concat(df)

    if summarize:
        fn = op.join(target_dir, f'group_source-{source}_fixations_summary.tsv')
    else:
        fn = op.join(target_dir, f'group_source-{source}_fixations.tsv')

    df.to_csv(fn, sep='\t')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('source', default=None)
    parser.add_argument('--no_summary', dest='summarize', action='store_false')
    args = parser.parse_args()

    main(args.source, args.summarize)
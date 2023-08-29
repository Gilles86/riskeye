import re
from io import StringIO
import argparse
import os.path as op
import os
import subprocess
import gzip

import glob
import pandas as pd
import shutil

runs = list(range(1,  6))

def get_subject_folder(subject, root_folder='/data'):
    candidate_dirs = glob.glob(op.join(root_folder, 'sourcedata', f'sub_{subject:02d}*'))
    assert(len(candidate_dirs) == 1)
    return candidate_dirs[0]
    
def extract_data(subject, root_folder='/data'):
    subject = int(subject)

    dir = get_subject_folder(subject, root_folder) 

    for run in runs:
        fn = op.join(dir, f'Rs{subject:02d}rn{run:02d}.edf')
        asc_fn = fn.replace('.edf', '.asc')
        gaze_target_fn = fn.replace('.edf', '.gaz.gz')

        # get gaze
        cmd = f'edf2asc  -t -y -z -v -s -vel {fn}'

        subprocess.run(cmd, shell=True)

        with open(asc_fn, 'rb') as asc_file, gzip.open(gaze_target_fn, 'wb') as target_file:
                target_file.writelines(asc_file)

        os.remove(asc_fn)

        # get messages
        cmd = f'edf2asc  -t -y -z -v -e {fn}'
        msg_target_fn = fn.replace('.edf', '.msg.gz')

        subprocess.run(cmd, shell=True)

        with open(asc_fn, 'rb') as asc_file, gzip.open(msg_target_fn, 'wb') as target_file:
                target_file.writelines(asc_file)

        os.remove(asc_fn)

def get_experimental_messages(subject, root_folder='/data'):
    subject = int(subject)
    dir = get_subject_folder(subject, root_folder)

    messages = []

    for run in runs:
        msg_fn = op.join(dir, f'Rs{subject:02d}rn{run:02d}.msg.gz')

        with gzip.open(msg_fn, 'rt') as mfd:
            message_string = mfd.read()

        message_strings = re.findall(re.compile('MSG\t(?P<timestamp>[0-9.]+)\t(?P<message>[0-9]+_.+)'), message_string)
        r = re.compile('([0-9]+)_(.+)')

        tmp = pd.DataFrame(message_strings, columns=['timestamp', 'message']).astype({'timestamp':int})#.set_index('timestamp').head(10)
        tmp['trial'] = tmp['message'].map(lambda x: r.match(x).group(1)).astype(int)
        tmp['type'] = tmp['message'].map(lambda x: r.match(x).group(2)).map(lambda x: 'response' if x.startswith('R') else x)

        block = 0
        for ix, row in tmp.iterrows():
            if (row.trial == 1) & (row.type == 'rfx'):
                block += 1
            tmp.at[ix, 'block'] = int(block)


        tmp['block'] = tmp['block'].astype(int)

        tmp = tmp.set_index(['block', 'trial', 'type'], drop=True)

        messages.append(tmp)

    return pd.concat(messages, keys=runs, names=['run'])



def get_saccades(subject, root_folder='/data'):
    subject = int(subject)
    dir = get_subject_folder(subject, root_folder)

    runs = list(range(1, 6))

    saccades = []

    for run in runs:
        msg_file = op.join(dir, f'Rs{subject:02d}rn{run:02d}.msg.gz')

        with gzip.open(msg_file, 'rt') as mfd:
            message_string = mfd.read()

        message_strings = re.findall(re.compile('ESACC\t(?P<info>.+)'), message_string)

        csvString = ('\n'.join(message_strings))

        s = pd.read_csv(StringIO(csvString), sep='\t', names=['eye', 'start_timestamp', 'end_timestamp', 'duration', 'start_x', 'start_y', 'end_x', 'end_y', 'amp', 'peak_velocity'], na_values=['.', '   .'],)
                            #    dtype={'start_y':float})

        s.index.name = 'n'

        saccades.append(s)

    saccades = pd.concat(saccades, keys=runs, names=['run'])

    return saccades

def main(subject, root_folder='/data'):

    target_dir = op.join(root_folder, 'derivatives', 'pupil', f'sub-{subject:02d}', 'func')
    if not op.exists(target_dir):
        os.makedirs(target_dir)

    extract_data(subject, root_folder)
    messages = get_experimental_messages(subject, root_folder)
    saccades = get_saccades(subject, root_folder)

    messages.to_csv(op.join(target_dir, f'sub-{subject:02d}_messages.tsv'), sep='\t')
    saccades.to_csv(op.join(target_dir, f'sub-{subject:02d}_saccades.tsv'), sep='\t')

    dir = get_subject_folder(subject, root_folder) 

    for run in runs:
        gaze_fn = op.join(dir, f'Rs{subject:02d}rn{run:02d}.gaz.gz')
        shutil.move(gaze_fn, op.join(target_dir, f'sub-{subject:02d}_run-{run}_gaze.tsv.gz'))

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('subject', type=int)
    argparser.add_argument('--root_folder',
            default='/data')
    args = argparser.parse_args()
    main(args.subject, args.root_folder)
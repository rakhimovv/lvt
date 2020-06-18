import argparse
import os
import shutil
import subprocess

import pandas as pd
import pytube
from joblib import Parallel
from joblib import delayed

"""
Script from https://github.com/piaxar/kinetics-downloader
"""

REQUIRED_COLUMNS = ['label', 'youtube_id', 'time_start', 'time_end', 'split', 'is_cc']
TRIM_FORMAT = '%06d'
URL_BASE = 'https://www.youtube.com/watch?v='

VIDEO_EXTENSION = '.mp4'
VIDEO_FORMAT = 'mp4'
TOTAL_VIDEOS = 0


def create_file_structure(path, folders_names):
    """
    Creates folders in specified path.
    :return: dict
        Mapping from label to absolute path folder, with videos of this label
    """
    mapping = {}
    if not os.path.exists(path):
        os.mkdir(path)
    for name in folders_names:
        dir_ = os.path.join(path, name)
        if not os.path.exists(dir_):
            os.mkdir(dir_)
        mapping[name] = dir_
    return mapping


def download_clip(row, label_to_dir, trim, count):
    """
    Download clip from youtube.
    row: dict-like objects with keys: ['label', 'youtube_id', 'time_start', 'time_end']
    'time_start' and 'time_end' matter if trim is True
    trim: bool, trim video to action ot not
    """

    label = row['label']
    filename = row['youtube_id']
    time_start = row['time_start']
    time_end = row['time_end']

    start = str(time_start)
    end = str(time_end - time_start)

    # if trim, save full video to tmp folder
    output_path = label_to_dir['tmp'] if trim else label_to_dir[label]
    output_filename = os.path.join(label_to_dir[label],
                                   filename + '_{}_{}'.format(start, end) + VIDEO_EXTENSION)

    # don't download if already exists
    if not os.path.exists(output_filename):
        print('Start downloading: ', filename)
        try:
            pytube.YouTube(URL_BASE + filename). \
                streams.filter(subtype=VIDEO_FORMAT).first(). \
                download(output_path, filename)
            print('Finish downloading: ', filename)
        except KeyError:
            print('Unavailable video: ', filename)
            return
        except Exception as e:
            print('Don\'t know why something went wrong. ID=%s' % filename)
            print(e)
            return
    else:
        print('Already downloaded: ', filename)

    if trim:
        # Take video from tmp folder and put trimmed to final destination folder
        # better write full path to video

        input_filename = os.path.join(output_path, filename + VIDEO_EXTENSION)

        if os.path.exists(output_filename):
            print('Already trimmed: ', filename)
        else:
            print('Start trimming: ', filename)
            # Construct command to trim the videos (ffmpeg required).
            command = 'ffmpeg -i "{input_filename}" ' \
                      '-ss {time_start} ' \
                      '-t {time_end} ' \
                      '-c:v libx264 -c:a copy -threads 1 ' \
                      '"{output_filename}"'.format(
                input_filename=input_filename,
                time_start=start,
                time_end=end,
                output_filename=output_filename
            )
            try:
                subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT)
            except subprocess.CalledProcessError:
                print('Error while trimming: ', filename)
                return False
            print('Finish trimming: ', filename)

    print('Processed %i out of %i' % (count + 1, TOTAL_VIDEOS))


def main(input_csv, output_dir, trim, num_jobs):
    global TOTAL_VIDEOS

    assert input_csv[-4:] == '.csv', 'Provided input is not a .csv file'
    links_df = pd.read_csv(input_csv)
    assert all(elem in REQUIRED_COLUMNS for elem in links_df.columns.values), \
        'Input csv doesn\'t contain required columns.'

    # Creates folders where videos will be saved later
    # Also create 'tmp' directory for temporary files
    folders_names = links_df['label'].unique().tolist() + ['tmp']
    label_to_dir = create_file_structure(path=output_dir,
                                         folders_names=folders_names)

    TOTAL_VIDEOS = links_df.shape[0]
    # Download files by links from dataframe
    Parallel(n_jobs=num_jobs)(
        delayed(download_clip)(row, label_to_dir, trim, count) for count, row in links_df.iterrows())

    # Clean tmp directory
    shutil.rmtree(label_to_dir['tmp'])


if __name__ == '__main__':
    description = 'Script for downloading and trimming videos from Kinetics dataset.' \
                  'Supports Kinetics-400 as well as Kinetics-600.'
    p = argparse.ArgumentParser(description=description)
    p.add_argument('input_csv', type=str,
                   help=('Path to csv file, containing links to youtube videos.\n'
                         'Should contain following columns:\n'
                         'label, youtube_id, time_start, time_end, split, is_cc'))
    p.add_argument('output_dir', type=str,
                   help='Output directory where videos will be saved.\n'
                        'It will be created if doesn\'t exist')
    p.add_argument('--trim', action='store_true', dest='trim', default=False,
                   help='If specified, trims downloaded video, using values, provided in input_csv.\n'
                        'Requires "ffmpeg" installed and added to environment PATH')
    p.add_argument('--num-jobs', type=int, default=1,
                   help='Number of parallel processes for downloading and trimming.')
    main(**vars(p.parse_args()))

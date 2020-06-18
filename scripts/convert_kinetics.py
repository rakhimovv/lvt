import argparse
import os
import subprocess
from glob import glob
from os.path import join as join_path
from shutil import rmtree

import PIL
from PIL import Image
from joblib import Parallel
from joblib import delayed
import numpy as np
import pandas as pd

TOTAL_VIDEOS = 0


def process_video(path, output_dir, img_size, count):
    head, name = os.path.split(path)
    cls = head.split('/')[-1]

    save_dir = join_path(output_dir, cls, name.split('.')[0])
    save_path = join_path(save_dir, '%d.png')

    log_name = '/'.join(save_dir.split('/')[-2:])

    if os.path.exists(save_dir):
        rmtree(save_dir)

    os.makedirs(save_dir)
    print('Start converting: ', log_name)
    # Construct command to convert the videos (ffmpeg required).
    command = 'ffmpeg -threads 1 -i "{input_filename}" ' \
              '"{save_path}"'.format(
        input_filename=path,
        save_path=save_path
    )
    try:
        subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        print('Error while converting: ', log_name)
        print(e)
        print(e.output)
        with open("fail_convert_256_test.log", "a") as f:
            f.write(path + '\n  ')
    print('Finish converting: ', log_name)

    frames = glob(join_path(save_dir, '*.png'))

    for frame in frames:
        img = Image.open(frame)
        width, height = img.size
        new_dim = min(width, height)

        left = (width - new_dim) / 2
        top = (height - new_dim) / 2
        right = (width + new_dim) / 2
        bottom = (height + new_dim) / 2

        img = img.crop((left, top, right, bottom))
        img = img.resize((img_size, img_size), resample=PIL.Image.LANCZOS)
        img.save(frame, 'PNG')

    print('Processed %i out of %i' % (count + 1, TOTAL_VIDEOS))


def main(lables, video_dir, output_dir, img_size, num_jobs, input_csv):
    print('Start kinetics convert script')
    global TOTAL_VIDEOS

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # print(join_path(video_dir, '*'))
    # print(glob(join_path(video_dir, '*')))

    videos = glob(join_path(video_dir, '*', '*.mp4'))

    if input_csv:
        ids = list(map(lambda x: '_'.join(x.split('/')[-1].split('.')[0].split('_')[:-2]), videos))
        # rem_ids = list(map(lambda x: '_'.join(x.split('_')[:-2]), REM_IDS))

        links_df = pd.read_csv(input_csv)
        remaining = links_df[np.logical_not(links_df['youtube_id'].isin(ids))]
        print('Remaining,', len(remaining))

        remaining.to_csv('remaining_last.csv', index=None)
        return

    if lables:
        classes = np.loadtxt(lables, dtype=str, delimiter=',')
        videos = list(filter(lambda x: x.split('/')[-2] in classes, videos))

    TOTAL_VIDEOS = len(videos)
    print('Total number of videos:', TOTAL_VIDEOS)

    Parallel(n_jobs=num_jobs)(
        delayed(process_video)(path, output_dir, img_size, count) for count, path in enumerate(videos))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--lables', type=str, default=None, help='Path to labels for processing')
    p.add_argument('--video_dir', type=str, help='Dir with the downloaded videos')
    p.add_argument('--output_dir', type=str, required=True, help='Output directory where frames will be saved.')
    p.add_argument('--img_size', type=int, default=64, help='Size of target frames')
    p.add_argument('--num_jobs', type=int, default=1,
                   help='Number of parallel processes for processing')
    p.add_argument('--input_csv', type=str, default=None, help='Size of target frames')
    main(**vars(p.parse_args()))

# example:
# python convert_kinetics.py --video_dir ~/eccv/rus/datasets/kinetics600/train
# --output_dir ~/eccv/rus/datasets/kinetics600/train_frames --num_jobs 5 --img_size 64

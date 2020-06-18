import os

import numpy as np

from vidgen.data import DatasetCatalog, MetadataCatalog
from vidgen.utils.lables import KINETICS_LABEL_IDX, KINETICS_IDX_LABEL
from vidgen.utils.strings import natural_sorted


def get_latent_video_paths(root, use_cache=True):
    assert os.path.isdir(root) or os.path.islink(root), f'{root} is not a valid directory'

    video_paths = []
    cache_path = os.path.join(root, 'latent_video_paths.npy')
    if use_cache and os.path.exists(cache_path):
        video_paths = np.load(cache_path, allow_pickle=True).tolist()
        return video_paths

    video_idx = 0
    for root, dirs, files in os.walk(root):
        if len(dirs) > 0:
            # video folder must contain only npy files
            continue
        is_video = True
        files = natural_sorted(files)
        image_paths = []
        for file in files:
            if not file.endswith('.npy'):
                is_video = False
                break
            else:
                image_paths.append(os.path.join(root, file))
        if is_video:
            video_paths.append({"video_path": root, "latent_paths": image_paths, "video_idx": video_idx})
            video_idx += 1

    if use_cache and not os.path.exists(cache_path):
        np.save(cache_path, video_paths)

    return video_paths


def get_kinetics_video_paths(root, use_cache=True, filter=None):
    assert os.path.isdir(root) or os.path.islink(root), f'{root} is not a valid directory'

    video_paths = []
    cache_path = os.path.join(root, 'latent_video_paths.npy')

    if use_cache and os.path.exists(cache_path):
        video_paths = np.load(cache_path, allow_pickle=True).tolist()
        if filter is None:
            return video_paths
        else:
            filtered_video_paths = []
            for i in range(len(video_paths)):
                if KINETICS_IDX_LABEL[video_paths[i]['class']] in filter:
                    filtered_video_paths.append(video_paths[i])
            return filtered_video_paths

    video_idx = 0
    for root, dirs, files in os.walk(root):
        if len(dirs) > 0:
            # video folder must contain only npy files
            continue
        is_video = True
        files = natural_sorted(files)
        latent_names = []
        for file in files:
            if not file.endswith('.npy'):
                is_video = False
                break
            else:
                latent_names.append(file)
        if is_video:
            d = {"video_root": root, "latent_names": latent_names,
                 "video_idx": video_idx,
                 "class": KINETICS_LABEL_IDX[root.split('/')[-2]]
                 }
            # print(d)
            print(video_idx)
            video_paths.append(d)
            video_idx += 1

    if use_cache and not os.path.exists(cache_path):
        np.save(cache_path, video_paths)

    if filter is None:
        return video_paths
    else:
        filtered_video_paths = []
        for i in range(len(video_paths)):
            if KINETICS_IDX_LABEL[video_paths[i]['class_idx']] in filter:
                filtered_video_paths.append(video_paths[i])
        return filtered_video_paths


def register_latents(name, root):
    DatasetCatalog.register(name, lambda: get_latent_video_paths(root))
    MetadataCatalog.get(name).set(root=root)


def register_kinetics_latents(name, root, filter=None):
    DatasetCatalog.register(name, lambda: get_kinetics_video_paths(root, filter=filter))
    MetadataCatalog.get(name).set(root=root)

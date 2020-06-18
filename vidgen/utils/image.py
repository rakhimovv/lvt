import glob
import os

import numpy as np
import torch.utils.data as data
from PIL import Image, ImageOps
from fvcore.common.file_io import PathManager

from vidgen.utils.lables import KINETICS_LABEL_IDX
from vidgen.utils.strings import natural_sorted


def make_grid(imgs, nrow=8):
    """
    Code borrowed from
    https://stackoverflow.com/questions/26521365/cleanly-tile-numpy-array-of-images-stored-in-a-flattened-1d-format/26521997
    """

    # Padding
    if imgs.shape[0] % nrow == 0:
        row_padding = 0
    else:
        row_padding = nrow - imgs.shape[0] % nrow
    if row_padding > 0:
        imgs = np.concatenate([imgs, np.zeros((row_padding, *imgs.shape[1:]), dtype=imgs.dtype)], axis=0)

    # Tiling Loop (The conditionals are not necessary anymore)
    tiled = []
    for i in range(0, imgs.shape[0], nrow):
        tiled.append(np.concatenate([imgs[j] for j in range(i, i + nrow)], axis=1))

    tiled = np.concatenate(tiled, axis=0)
    return tiled


def tensor2im(image_tensor, imtype=np.uint8, normalize=True, tile=False, nrow=8):
    """
    Converts a Tensor into a Numpy array

    Args:
        image_tensor: image tensor
        imtype: the desired type of the converted numpy array
        normalize: the number of images displayed in each row of the grid
        tile: create a grid from images

    Returns:
        np.array
    """
    if isinstance(image_tensor, list):
        image_numpy = []
        for i in range(len(image_tensor)):
            image_numpy.append(tensor2im(image_tensor[i], imtype, normalize))
        return image_numpy

    if image_tensor.dim() == 4:
        # transform each image in the batch
        images_np = []
        for b in range(image_tensor.size(0)):
            one_image = image_tensor[b]
            one_image_np = tensor2im(one_image, imtype, normalize)
            images_np.append(one_image_np.reshape(1, *one_image_np.shape))
        images_np = np.concatenate(images_np, axis=0)
        if tile:
            images_tiled = make_grid(images_np, nrow)
            return images_tiled
        else:
            return images_np

    if image_tensor.dim() == 2:
        image_tensor = image_tensor.unsqueeze(0)
    image_numpy = image_tensor.detach().cpu().float().numpy()
    if normalize:
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    else:
        image_numpy = np.transpose(image_numpy, (1, 2, 0)) * 255.0
    image_numpy = np.clip(image_numpy, 0, 255)

    if image_numpy.shape[2] == 1:
        image_numpy = image_numpy[:, :, 0]
    return image_numpy.astype(imtype)


def tensor2labelgray(label_tensor, n_label, imtype=np.uint8, tile=False, nrow=8):
    """
    Converts a one-hot tensor into a gray label map
    """
    if label_tensor.dim() == 4:
        # transform each image in the batch
        images_np = []
        for b in range(label_tensor.size(0)):
            one_image = label_tensor[b]
            one_image_np = tensor2labelgray(one_image, n_label, imtype)
            images_np.append(one_image_np.reshape(1, *one_image_np.shape))
        images_np = np.concatenate(images_np, axis=0)
        if tile:
            images_tiled = make_grid(images_np, nrow)
            return images_tiled
        else:
            images_np = images_np[0]
            return images_np

    if label_tensor.dim() == 1:
        return np.zeros((64, 64), dtype=np.uint8)
    if n_label == 0:
        return tensor2im(label_tensor, imtype)
    label_tensor = label_tensor.cpu().float()
    if label_tensor.size()[0] > 1:
        label_tensor = label_tensor.max(0, keepdim=True)[1]
    label_numpy = np.transpose(label_tensor.numpy(), (1, 2, 0))
    result = label_numpy.astype(imtype)
    return result


def save_image(image_numpy, image_path, create_dir=False):
    if create_dir:
        os.makedirs(os.path.dirname(image_path), exist_ok=True)
    if len(image_numpy.shape) == 2:
        image_numpy = np.expand_dims(image_numpy, axis=2)
    if image_numpy.shape[2] == 1:
        image_numpy = np.repeat(image_numpy, 3, 2)
    image_pil = Image.fromarray(image_numpy)

    # save to png
    image_pil.save(image_path.replace('.jpg', '.png'))


IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tiff', '.webp'
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def get_image_paths(root, use_cache=True, is_kinetics=False):
    assert os.path.isdir(root) or os.path.islink(root), f'{root} is not a valid directory'

    image_paths = []
    cache_path = os.path.join(root, 'image_paths.npy')

    if use_cache and os.path.exists(cache_path):
        image_paths = np.load(cache_path, allow_pickle=True).tolist()
        return image_paths

    for root, _, files in os.walk(root):
        for file in natural_sorted(files):
            if is_image_file(file):
                if file.startswith('._'):
                    # https://apple.stackexchange.com/questions/14980/why-are-dot-underscore-files-created-and-how-can-i-avoid-them
                    print(f"skipping {file}")
                else:
                    path = os.path.join(root, file)
                    d = {"image_path": path}
                    if is_kinetics:
                        d['class'] = KINETICS_LABEL_IDX[root.split('/')[-2]]
                    image_paths.append(d)

    if use_cache:
        try:
            np.save(cache_path, image_paths)
        except PermissionError:
            print('PermissionError')

    return image_paths


def get_video_paths(root, use_cache=True, is_kinetics=False):
    assert os.path.isdir(root) or os.path.islink(root), f'{root} is not a valid directory'

    video_paths = []
    cache_path = os.path.join(root, 'video_paths.npy')

    if use_cache and os.path.exists(cache_path):
        video_paths = np.load(cache_path, allow_pickle=True).tolist()
        return video_paths

    video_idx = 0
    for root, dirs, files in os.walk(root):
        if len(dirs) > 0:
            # video folder must contain only images
            continue
        is_video = True
        image_names = []
        for file in natural_sorted(files):
            if not is_image_file(file):
                is_video = False
                break
            else:
                if not file.startswith('._'):
                    # https://apple.stackexchange.com/questions/14980/why-are-dot-underscore-files-created-and-how-can-i-avoid-them
                    image_names.append(file)
        if is_video and len(image_names) > 0:
            # d = {"video_path": root, "image_paths": image_paths, "video_idx": video_idx}
            d = {"video_root": root, "image_names": image_names, "video_idx": video_idx}
            if is_kinetics:
                d['class'] = KINETICS_LABEL_IDX[root.split('/')[-2]]
            video_paths.append(d)
            video_idx += 1

    if use_cache:
        try:
            np.save(cache_path, video_paths)
        except PermissionError:
            print('PermissionError')

    return video_paths


def read_image(file_name, format=None):
    """
    Read an image into the given format.
    Will apply rotation and flipping if the image has such exif information.

    Args:
        file_name (str): image file path
        format (str): one of the supported image modes in PIL, or "BGR"

    Returns:
        image (np.ndarray): an HWC image
    """
    with PathManager.open(file_name, "rb") as f:
        image = Image.open(f)

        # capture and ignore this bug: https://github.com/python-pillow/Pillow/issues/3973
        try:
            image = ImageOps.exif_transpose(image)
        except Exception:
            pass

        if format is not None:
            # PIL only supports RGB, so convert to RGB and flip channels over below
            conversion_format = format
            if format == "BGR":
                conversion_format = "RGB"
            image = image.convert(conversion_format)
        image = np.asarray(image)
        if format == "BGR":
            # flip channels if needed
            image = image[:, :, ::-1]
        # PIL squeezes out the channel dimension for "L", so make it HWC
        if format == "L":
            image = np.expand_dims(image, -1)
        return image


def glob_file_list(root):
    return sorted(glob.glob(os.path.join(root, '*')))


def default_loader(path):
    return Image.open(path).convert('RGB')


class ImageFolder(data.Dataset):

    def __init__(self, root, transform=None, return_paths=False,
                 loader=default_loader):
        imgs = get_image_paths(root)
        if len(imgs) == 0:
            raise (RuntimeError(f"Found 0 images in: {root}\nSupported image extensions are: {IMG_EXTENSIONS}"))

        self.root = root
        self.imgs = imgs
        self.transform = transform
        self.return_paths = return_paths
        self.loader = loader

    def __getitem__(self, index):
        path = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.return_paths:
            return img, path
        else:
            return img

    def __len__(self):
        return len(self.imgs)

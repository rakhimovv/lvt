import argparse
import os

import numpy as np
import torch
from fvcore.common.checkpoint import Checkpointer
from fvcore.common.file_io import PathManager

from vidgen.config import get_cfg
from vidgen.engine import default_setup
from vidgen.modeling.meta_arch import build_model
from vidgen.utils.image import save_image, get_image_paths, read_image


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    default_setup(cfg, args)
    return cfg


def load_video(video_dir, scale_to_zeroone=True):
    """
    Args:
        video_dir: path to folder with priming images
        scale_to_zeroone: scale the image by 255 or not
    """
    img_paths = [x['image_path'] for x in get_image_paths(video_dir, use_cache=False)]
    video = [np.ascontiguousarray(read_image(img_path)).transpose(2, 0, 1) for img_path in img_paths]
    video = np.stack(video, axis=0).astype('float32')
    if scale_to_zeroone:
        video /= 255.
    return video


def save_video(video, output_dir):
    """
    Save video
    Args:
        video: shape [T, H, W, C]
        video_dir: save video under video_dir/sample/
    """
    PathManager.mkdirs(output_dir)
    for frame_idx in range(len(video)):
        frame_path = os.path.join(output_dir, f'{frame_idx}.png')
        save_image(video[frame_idx], frame_path)


@torch.no_grad()
def sample_videos(args):
    # load config
    cfg = setup(args)
    cfg.TEST.EVALUATORS = "VTSampler"
    cfg.TEST.NUM_SAMPLES = 1

    # load videotransformer
    vt = build_model(cfg)
    Checkpointer(vt.model).resume_or_load(cfg.MODEL.GENERATOR.WEIGHTS, resume=False)
    vt.eval()

    # load vqvae
    vq_cfg = get_cfg()
    vq_cfg.merge_from_file(cfg.TEST.VT_SAMPLER.VQ_VAE.CFG)
    vqvae = build_model(vq_cfg)
    Checkpointer(vqvae.encoder).resume_or_load(cfg.TEST.VT_SAMPLER.VQ_VAE.ENCODER_WEIGHTS, resume=False)
    Checkpointer(vqvae.generator).resume_or_load(cfg.TEST.VT_SAMPLER.VQ_VAE.GENERATOR_WEIGHTS, resume=False)
    Checkpointer(vqvae.codebook).resume_or_load(cfg.TEST.VT_SAMPLER.VQ_VAE.CODEBOOK_WEIGHTS, resume=False)
    vqvae.eval()

    # load data
    scale_to_zeroone = vq_cfg.INPUT.SCALE_TO_ZEROONE
    n_prime = cfg.TEST.VT_SAMPLER.N_PRIME
    images = load_video(args.video_dir)[:n_prime]  # (T, C, H, W) = (n_prime, 3, 64, 64)
    assert images.shape == (n_prime, 3, 64, 64)
    print(f"Loaded {n_prime} priming frames")

    # sample
    latent_sequence = vqvae([{'image_sequence': images}])[0]['latent']  # (n_prime, nc, h, w) = (n_prime, nc, 16, 16)
    print(f"Transferred to latent codes.")
    _, nc, h, w = latent_sequence.shape
    new_sequence = latent_sequence.new_zeros(16, nc, h, w)
    new_sequence[:n_prime] = latent_sequence
    samples = vt([{'image_sequence': new_sequence}])[0]['samples']  # list of samples
    print(f"Sampled new video.")
    sample = samples[0].squeeze(0)  # T, h, w if nc == 1 or nc, T, h, w
    if sample.dim() == 4:
        sample = sample.transpose(0, 1)  # T, nc, h, w
    sample = vqvae.decode(sample)  # T, 3, H, W
    sample = vqvae.back_normalizer(sample)  # T, 3, H, W
    if scale_to_zeroone:
        sample = sample * 255
    sample.clamp_(0.0, 255.0)
    sample = sample.permute(0, 2, 3, 1).contiguous()  # T, H, W, 3
    sample = sample.detach().cpu().numpy().astype(np.uint8)
    save_video(sample, cfg.OUTPUT_DIR)
    print(f"Saved new video.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sample video with 16 frames given priming frames")
    parser.add_argument("--config-file", required=True, metavar="FILE", help="path to config file")
    parser.add_argument("--video-dir", required=True, help="path to video folder")
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()
    print("Command Line Args:", args)
    sample_videos(args)

# -*- coding: utf-8 -*-
"""
This file registers pre-defined datasets at hard-coded paths, and their metadata.

We hard-code some paths to the dataset that's assumed to
exist in "./datasets/".
"""

import os

from .bair import register_bair
from .kinetics import register_kinetics
from .latents import register_latents, register_kinetics_latents


def register_all_bair(root="datasets"):
    SPLITS = [
        ("bair_train", "bair", "train", True),
        ("bair_train_seq", "bair", "train", False),
        ("bair_test", "bair", "test", True),
        ("bair_test_seq", "bair", "test", False),
    ]
    for name, dirname, phase, load_images in SPLITS:
        register_bair(name, os.path.join(root, dirname), phase, load_images)


def register_all_kinetics(root="datasets"):
    SPLITS = [
        ("kinetics_train", "kinetics600", "train", True),
        ("kinetics_train_seq", "kinetics600", "train", False),
        ("kinetics_test", "kinetics600", "test", True),
        ("kinetics_test_seq", "kinetics600", "test", False),
        ("kinetics_train256", "kinetics600", "train256", True),
        ("kinetics_train256_seq", "kinetics600", "train256", False),
        ("kinetics_test256", "kinetics600", "test256", True),
        ("kinetics_test256_seq", "kinetics600", "test256", False),
    ]
    for name, dirname, phase, load_images in SPLITS:
        register_kinetics(name, os.path.join(root, dirname), phase, load_images)


# Register them all under "./datasets"
register_all_bair()
register_all_kinetics()

register_latents("prdvqvae_train", "datasets/prdvqvae2/inference/bair_train_seq")
register_latents("prdvqvae_test", "datasets/prdvqvae2/inference/bair_test_seq")

register_kinetics_latents("kdvqvae_train", "datasets/K-DVQVAE/inference/kinetics_train_seq")
register_kinetics_latents("kdvqvae_test", "datasets/K-DVQVAE/inference/kinetics_test_seq")

# Latent Video Transformer

Code for paper _Latent Video Transformer_.


## Preparation

The training routine code is based on [detectron2](https://github.com/facebookresearch/detectron2).

Run this command after cloning the repository.

```bash
python setup.py build develop
```


## Inference on the pretrained model

In order to run inference use the command:

```bash
CUDA_VISIBLE_DEVICES=<gpus> python scripts/generate_videos.py --video-dir ./example --config-file configs/vt/DSFVT.yaml MODEL.GENERATOR.WEIGHTS pretrained/DSFVT/netG/model_final.pth OUTPUT_DIR ./example/sample
```

It takes the following parameters:
* video-dir — Folder containing priming frames.
* config-file — Config file for specific type of LVT model
* any other parameters insided config-file you would like to change


## Datasets

### Bair

Download the dataset:

```bash
wget http://rail.eecs.berkeley.edu/datasets/bair_robot_pushing_dataset_v0.tar -P ./bair
tar -xvf ./bair/bair_robot_pushing_dataset_v0.tar -C ./bair
```

Preprocess the dataset:

```bash
python ./scripts/convert_bair.py --data_dir ./bair
```


### Kinetics-600

Kinetics-600 dataset is presented as a set of links to YouTube videos.

Download links:
```bash
mkdir ./kinetics/
wget https://storage.googleapis.com/deepmind-media/Datasets/kinetics600.tar.gz -P ./kinetics/
tar -xvf ./kinetics/kinetics600.tar.gz -C ./kinetics/
rm ./kinetics/kinetics600.tar.gz
```

Download data from YouTube:

```bash
python  ./scripts/download_kinetics.py ./kinetics/kinetics600/train.csv ./kinetics//kinetics600/train_vid --trim --num-jobs 1
python  ./scripts/download_kinetics.py ./kinetics/kinetics600/test.csv ./kinetics/kinetics600/test_vid --trim --num-jobs 1
```

Note, that YouTube can block you from downloading videos. That is why it is important not to load many videos simultaneously.


Preprocessing of videos includes:
1. Trimming videos to the scecified 10-sec range
2. Converting videos to png files
3. Center-crop each image 

```bash
python ./scripts/convert_kinetics.py --video_dir ./kinetics/kinetics600/train --output_dir ./kinetics/kinetics600/train_frames --num_jobs 5 --img_size 64
python ./scripts/convert_kinetics.py --video_dir ./kinetics/kinetics600/test --output_dir ./kinetics/kinetics600/test_frames --num_jobs 5 --img_size 64
```

Preprocessing script will store images in train_frames and test_frames folders.



## VQVAE

### Training

In order to train VQVAE run the following command. If you want to modify some parameters, consider changing them in the config _configs/vqvae/PR-DVQVAE2.yaml_.

```bash
CUDA_VISIBLE_DEVICES=<gpus> python tools/train_net.py --config-file configs/vqvae/PR-DVQVAE2.yaml --num-gpus <number of gpus> OUTPUT_DIR experiments/PR-DVQVAE2
```

### Codes sampling

After training of VQVAE one should run code extraction on train data:

```bash
CUDA_VISIBLE_DEVICES=<gpus> python tools/train_net.py --eval-only --config-file configs/vqvae/PR-DVQVAE2.yaml OUTPUT_DIR experiments/PR-DVQVAE2 TEST.EVALUATORS "CodesExtractor" DATASETS.TEST "kinetics_train_seq"
```


## Train Latent Transformer

Latent transformer is trained on codes extracted with VQVAE. You should run Latent Transformer after VQVAE training finished. 

Note, that in the config file, you should specify the dataset for latent codes:

```yaml
DATASETS:
  TRAIN: ("prdvqvae_train",)
  TEST: ("prdvqvae_test",)
```

In order to specify path to codes, modify file vidgen/data/datasets/builtin.py:

```python
register_latents("prdvqvae_train", "datasets/prdvqvae2/inference/bair_train_seq")
register_latents("prdvqvae_test", "datasets/prdvqvae2/inference/bair_test_seq")

register_kinetics_latents("kdvqvae_train", "datasets/K-DVQVAE/inference/kinetics_train_seq")
register_kinetics_latents("kdvqvae_test", "datasets/K-DVQVAE/inference/kinetics_test_seq")
```


```bash
CUDA_VISIBLE_DEVICES=<gpus> python tools/train_net.py --config-file configs/vt/DSFVT.yaml --num-gpus 1 OUTPUT_DIR experiments/vt/DSFVT 
```




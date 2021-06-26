# NeX: Real-time View Synthesis with Neural Basis Expansion

### [Project Page](https://nex-mpi.github.io/) | [Video](https://www.youtube.com/watch?v=HyfkF7Z-ddA) | [Paper](https://arxiv.org/pdf/2103.05606.pdf) | [COLAB](https://colab.research.google.com/drive/1hXVvYdAwLA0EFg2zrafJUE0bFgB_F7PU#scrollTo=TFbN4mrJCp8o&sandboxMode=true) | [Shiny Dataset](https://vistec-my.sharepoint.com/:f:/g/personal/pakkapon_p_s19_vistec_ac_th/EnIUhsRVJOdNsZ_4smdhye0B8z0VlxqOR35IR3bp0uGupQ?e=TsaQgM)

[![Open NeX in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1hXVvYdAwLA0EFg2zrafJUE0bFgB_F7PU#scrollTo=TFbN4mrJCp8o&sandboxMode=true)

[![NeX](https://i.imgur.com/TfXtYdC.png)](https://www.youtube.com/watch?v=HyfkF7Z-ddA)

We present NeX, a new approach to novel view synthesis based on enhancements of multiplane image (MPI) that can reproduce NeXt-level view-dependent effects---in real time. Unlike traditional MPI that uses a set of simple RGBα planes, our technique models view-dependent effects by instead parameterizing each pixel as a linear combination of basis functions learned from a neural network. Moreover, we propose a hybrid implicit-explicit modeling strategy that improves upon fine detail and produces state-of-the-art results. Our method is evaluated on benchmark forward-facing datasets as well as our newly-introduced dataset designed to test the limit of view-dependent modeling with significantly more challenging effects such as the rainbow reflections on a CD. Our method achieves the best overall scores across all major metrics on these datasets with more than 1000× faster rendering time than the state of the art.
## Table of contents
-----
  * [TL;DR](#Getting-started)
  * [Installation](#Installation)
  * [Dataset](#Dataset)
  * [Training](#Training)
  * [Rendering](#Rendering)
  * [Citation](#citation)
------

## Getting started

```shell
conda env create -f environment.yml
./download_demo_data.sh
conda activate nex
python train.py -scene data/crest_demo -model_dir crest -http
tensorboard --logdir runs/
```

## Installation
We provide `environment.yml` to help you setup a conda environment. 

```shell
conda env create -f environment.yml
```

## Dataset

Running NeX on your own images. You need to install [COLMAP](https://colmap.github.io/) on your machine.

Then, put your images into a directory following this structure
```
<scene_name>
|-- images
     | -- image_name1.jpg
     | -- image_name2.jpg
     ...
```

The training code will automatically prepare a scene for you. You may have to tune `planes.txt` to get better reconstruction (see [dataset explaination](https://vistec-my.sharepoint.com/:t:/g/personal/pakkapon_p_s19_vistec_ac_th/EYBtE-X95pFLscoLFehUMtQBjrrYKQ9mxVEzKzNlDuoZLw?e=bODHZ4))


## Training

Run with the paper's config
```shell
python train.py -scene ${PATH_TO_SCENE} -model_dir ${MODEL_TO_SAVE_CHECKPOINT} -http
```

This implementation uses [scikit-image](https://scikit-image.org/) to resize images during training by default. The results and scores in the paper are generated using OpenCV's resize function. If you want the same behavior, please add `-cv2resize` argument.

Note that this code is tested on an Nvidia V100 32GB and 4x RTX 2080Ti GPU.

For reproducing our final result, you can run using the following command:
```shell
python train.py -scene data/[red/gray/black] -model_dir [red/gray/black] -layers 24 -sublayers 6 -epochs 1 -offset 80 -tb_toc 1 -hidden 128 -pos_level 7 -depth_level 7 -tb_saveimage 2 -num_workers 2 -llff_width 250 -web_width=4096
```
Note that when your GPU runs ouut of memeory, you can try reducing the number of layers, sublayers, and sampled rays.

## Rendering

The generated video result will be at `runs/video_output/[model_dir]/video.mp4`

### Video rendering

To generate a video that matches the real forward-facing rendering path, add `-nice_llff` argument, or `-nice_shiny` for shiny dataset



## Citation

```
@inproceedings{Wizadwongsa2021NeX,
    author = {Wizadwongsa, Suttisak and Phongthawee, Pakkapon and Yenphraphai, Jiraphon and Suwajanakorn, Supasorn},
    title = {NeX: Real-time View Synthesis with Neural Basis Expansion},
    booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)}, 
    year = {2021},
}
```

## Visit us 🦉
[![Vision & Learning Laboratory](https://i.imgur.com/hQhkKhG.png)](https://vistec.ist/vision) [![VISTEC - Vidyasirimedhi Institute of Science and Technology](https://i.imgur.com/4wh8HQd.png)](https://vistec.ist/)

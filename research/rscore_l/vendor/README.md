<p align="center">
    <!-- <img alt="scrstudio" src="media/sdf_studio_4.png" width="300"> -->
    <h1 align="center">SCRStudio <br> A Unified Framework for Scene Coordinate Regression</h1>
        <h3 align="center"><a href="https://drive.google.com/file/d/1yZw_ZeZkq6MjIZhTXfmQSPbVg8oohdKs/view?usp=sharing">Paper</a> 
    <img src="docs/module.png" center width="95%"/>
</p>

# About
SCRStudio is a unified and modular framework for Scene Coordinate Regression (SCR)-based visual localization, built on top of the [nerfstudio](https://github.com/nerfstudio-project/nerfstudio) project.

This library provides an interpretable and modular implementation of SCRs, breaking down components such as input encoding, network architecture, and supervision strategies. It offers a unified implementation of three major SCR methods: **ACE, GLACE, and R-SCoRe**. SCRStudio supports various pretrained local encodings (both sparse and dense) while incorporating state-of-the-art techniques for integrating global encodings.


# Quickstart

This guide will help you get started with the default R-SCoRe SCR model trained on the classic **Aachen** dataset.

## 1. Installation: Setup the Environment

### Create Environment

We recommend using **conda** to manage dependencies. Make sure to install [Conda](https://docs.conda.io/miniconda.html) before proceeding.


### Install Dependencies

Install PyTorch with CUDA (tested with CUDA 12.1 and 12.4). PyTorch Geometric and cuML are also required for encoding preprocessing.

For **CUDA 12.4**:
```bash
conda create -n scrstudio python=3.10 pytorch=2.5.1 torchvision=0.20.1 pytorch-cuda=12.4 cuml=25.02 -c pytorch  -c rapidsai -c conda-forge -c nvidia
conda activate scrstudio
pip install --upgrade pip
pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.5.1+cu124.html
```



### Install SCRStudio

```bash
git clone --recursive https://github.com/cvg/scrstudio.git
cd scrstudio
pip install --upgrade pip setuptools
pip install -e .
```

## 2. Train Your First Model

The following steps will train a **scrfacto** model, our recommended model for large scenes.

### Download the Data

```bash
# Download Aachen dataset:
scr-download-data aachen
# Download specific capture for NAVER Lab dataset:
scr-download-data naver --capture-name dept_1F
```

### Preprocessing for Training

**scrfacto** follows the methodology from **R-SCoRe** ([paper](https://arxiv.org/abs/2501.01421)), utilizing **PCA** for local encoding dimensionality reduction and **[Node2Vec](https://arxiv.org/abs/1607.00653)** for learning global encodings.

#### Local Encoding: PCA Compression
To reduce GPU memory usage for local encoding buffer, apply PCA on local encodings:
```bash
# Compute PCA for Dedode local encodings on Aachen dataset
scr-encoding-pca dedode --encoder.detector L --encoder.descriptor B --n_components 128 --data data/aachen
```

This saves the PCA components as PyTorch state dictionary at: `data/aachen/proc/pcad3LB_128.pth`

#### Global Encoding: Covisibility Graph & Node2Vec Training
Compute the pose overlap score for training images:

```bash
scr-overlap-score --data data/aachen/train --max_depth 50
```
This saves a sparse COO format overlap matrix at: `data/aachen/train/pose_overlap.npz`

Train a **Node2Vec** model on this graph:
```bash
scr-train node2vec --data data/aachen --pipeline.model.graph pose_overlap.npz --pipeline.model.edge_threshold 0.2
```
Use the trained global encoding:

```bash
cp outputs/aachen/node2vec/<timestamp>/scrstudio_models/head.pt data/aachen/train/pose_n2c.pt
```

### Model Training
Now, train the **scrfacto** model:
```bash
scr-train scrfacto --data data/aachen --pipeline.datamanager.train_dataset.feat_name pose_n2c.pt
```
Results are saved in `outputs/aachen/scrfacto/<timestamp>`.

### 3. Evaluation

#### Preprocessing for Evaluation

Compute NetVLAD retrieve features and compress them with Product Quantization (PQ):

```bash
scr-retrieval-feat --data data/aachen/train --pq
```
Results are saved in `data/aachen/train/netvlad_feats_pq.pkl`

### Running Evaluation
Compute retrieval features for test images and run evaluation:

```bash
scr-retrieval-feat --data data/aachen/test
scr-eval --load-config outputs/aachen/scrfacto/<timestamp>/config.yml --split test
```


# Publications

This code builds on previous camera relocalization pipelines, namely DSAC, DSAC++, DSAC*, ACE, GLACE, and R-SCoRe.
Please consider citing:

```
@inproceedings{brachmann2017dsac,
  title={{DSAC}-{Differentiable RANSAC} for Camera Localization},
  author={Brachmann, Eric and Krull, Alexander and Nowozin, Sebastian and Shotton, Jamie and Michel, Frank and Gumhold, Stefan and Rother, Carsten},
  booktitle={CVPR},
  year={2017}
}

@inproceedings{brachmann2018lessmore,
  title={Learning less is more - {6D} camera localization via {3D} surface regression},
  author={Brachmann, Eric and Rother, Carsten},
  booktitle={CVPR},
  year={2018}
}

@article{brachmann2021dsacstar,
  title={Visual Camera Re-Localization from {RGB} and {RGB-D} Images Using {DSAC}},
  author={Brachmann, Eric and Rother, Carsten},
  journal={TPAMI},
  year={2021}
}

@inproceedings{brachmann2023ace,
    title={Accelerated Coordinate Encoding: Learning to Relocalize in Minutes using RGB and Poses},
    author={Brachmann, Eric and Cavallari, Tommaso and Prisacariu, Victor Adrian},
    booktitle={CVPR},
    year={2023},
}

@inproceedings{wang2024glace,
    title={Glace: Global local accelerated coordinate encoding},
    author={Wang, Fangjinhua and Jiang, Xudong and Galliani, Silvano and Vogel, Christoph and Pollefeys, Marc},
    booktitle={CVPR},
    year={2024}
}

@inproceedings{jiang2025rscore,
      title={R-SCoRe: Revisiting Scene Coordinate Regression for Robust Large-Scale Visual Localization},
      author={Jiang, Xudong and Wang, Fangjinhua and Galliani, Silvano and Vogel, Christoph and Pollefeys, Marc},
      booktitle = {CVPR},
      year={2025}
}
```



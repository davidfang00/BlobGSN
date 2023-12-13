## BlobGSN: Generative Scene Networks with Mid-Level Blob Representations
**Unconstrained Scene Generation with Locally Conditioned Radiance Fields and Mid-Level Blob Representations**<br>
Allen Zhang, David Fang, Ilona Demler<br>

### [Project Page](https://ilonadem.github.io/blobgsn-demo/) | [Code](https://github.com/davidfang00/BlobGSN) | [Data](#datasets)

## Abstract 
We combined BlobGAN with Generative Scene Networks to generate editable 3D scenes. Namely, we use Gaussian "blobs" as input to generating a 2-D floorplan that is then used to locally condition a radiance field. The Gaussian blobs represent objects in a scene; by moving, shifting, scaling, removing, and adding the blobs in the latent space we are able to make corresponding changes in the rendered images. The result is a customizable and editable 3D scene, and a self-suprevised way of identifying and representing the objects in it.

## Requirements
This code was tested with Python 3.6 and CUDA 11.1.1, and uses Pytorch Lightning. A suitable conda environment named `blobgsn` can be created and activated with:
```
conda env create -f environment.yaml python=3.6
conda activate blobgsn
```

## Datasets
We use the Vizdoom and Replica datasets from Generative Scene Networks. They contain sequences of frames with the RGB and depth frames along with camera poses and intrinsics.

Dataset | Size | Download Link
--- | :---: | :---:
Vizdoom | 2.4 GB | [download](<https://docs-assets.developer.apple.com/ml-research/datasets/gsn/vizdoom.zip>)
Replica | 11.0 GB | [download](<https://docs-assets.developer.apple.com/ml-research/datasets/gsn/replica.zip>)

Datasets can be downloaded by running the following scripts:  
**VizDoom**<br>
```
python scripts/download_vizdoom.py
```
**Replica**<br>
```
python scripts/download_replica.py
```

## Interactive exploration demo
A Jupyter notebook is provided for interactive walkthroughs of scenes with the WASD keys based off of GSN. The notebook also allows for blob visualization and manipulation. The notebook interpolates the camera path to produce smooth trajectories.

Explore scene with WASD to set keypoints | Rendered trajectory
:---: | :---:
<img src="./assets/keyframes.gif" width=256px> | <img src="./assets/camera_trajectory.gif" width=256px>

## Training models
Download the training dataset and begin training with the following commands:  
**VizDoom**<br>
```
bash scripts/launch_gsn_vizdoom_64x64.sh
```

**Replica**<br>
```
bash scripts/launch_gsn_replica_64x64.sh
```

Training takes about 4 days to reach 500k iterations with a batch size of 8 on a single V100 GPU.

## Code Acknowledgements
Our code builds off of existing work:
- [BlobGAN](https://github.com/dave-epstein/blobgan)
- [Generative Scene Networks](https://apple.github.io/ml-gsn/)

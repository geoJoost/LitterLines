# LitterLines: An Annotated Dataset for Detection of Marine Litter Windrows in PlanetScope Imagery

<!-- let's add those when they are ready
[[`paper`](google.com)][[`demo`](google.com)][[`dataset`](google.com)]
-->

> LitterLines is a ready-to-analyze dataset for detecting marine litter windrows (MLWs) in PlanetScope imagery, consisting of 29 annotated scenes with over 2.4 million labeled pixels. Designed for machine learning applications, it enables the development and fine-tuning of models for MLW detection, supporting improved monitoring of marine litter from satellite-based sources. 

<p align="center">
  <img src="./doc/figures/litterlines_logo.png" alt="Logo 1" width="550">
  <img src="./doc/figures/overview_withMLW.png" alt="Logo 2" width="730">
</p>



## Getting Started

```
# install litterlines and its dependencies
pip install git+ssh://git@github.com/geoJoost/LitterLines.git

conda create --n litterlines

conda activate litterlines

# conda install pytorch <-- update with appropriate version 

conda env update --f environment.yml
```

## Dataloader output
*LitterLines* is dataset for marine litter detection, containing 1,016 image patches (256x256 pixels) from 966 line annotations. It uses the high-resolution PlanetScope imagery, with current implementation using the *Analytic Ortho Scene* products with TOA reflectance. The dataset's retrieval pipeline is modular, allowing for easy expansion and adaptation, including focusing on specific regions or sensor types.

Currently, a majority of the images are from Dove-C sensors (740 out of 1,016 images), suitable for models based on past detections. While the dataset may be too small for training deep neural networks from scratch, it is sufficient for fine-tuning pre-trained models and training random forest models, given its over 2.4 million positive MLW pixels.

To further illustrate the dataset's structure and its readiness for model training, the figure below presents a selection of 246-pixel image patches. Each patch is visualized using the RGB, NDVI, and RAI (Rotation-Absorption Index), with the final column displaying the corresponding annotation label.

<p align="center">
  <img src="./doc/figures/litterlines_patches.png" alt="LitterLinesvisualization" width="500">
</p>

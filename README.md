<p align=”center”>

<img width=”200" height=”200" src=”https://user-images.githubusercontent.com/75753187/123358567-aac7b900-d539-11eb-8275-0b380264bb4c.png" alt=”my banner”>

</p>



# Simulating the Effect of Sudden Temperature Increase in Winter on Plant Biomass - A BioMakerCA Project

![](https://github.com/Your_Repository_Name/Your_GIF_Name.gif)

This repository is made for the project described in "Simulating the effect of sudden temperature increase in winter on plant biomass" by Janneke Nouwen, Laura Stritzel, and Julian Roddeman.
This repository is based on the [self-organising-systems](https://github.com/google-research/self-organising-systems/tree/master) repository and see [this web article](https://google-research.github.io/self-organising-systems/2023/biomaker-ca/) or [this paper](https://arxiv.org/abs/2307.09320) for explanations of the base package.

## Introduction

Climate change is causing changes in ecosystems, leading to unusual plant behaviors like blooming in winter. This study focuses on the impact of rising winter temperatures on plant biomass.

The main research question is: **How do sudden warmer temperatures in winter influence plant biomass throughout the year?**

To answer this question, we created simulations using Python 3.11 and the Biomaker CA package, which employs Cellular Automata (CA) to model biomes. By simulating seasonal changes and introducing a sudden temperature increase in winter, we observe plant growth and reproduction to understand the impacts of an unusually warm winter. This repository includes all necessary scripts, configurations, and results for replicating our study.

## Install Requirements

1. Move into the package root.
2. Run `pip install -r requirements.txt`

## Install BioMakerCA

Install the [self-organising-systems](https://github.com/google-research/self-organising-systems/tree/master) package:

1. Download the .zip file from the biomaker-v1.0.0 release and unzip it.
2. Make sure to use Python 3.11.
3. (Optional) Create a conda environment and activate it.
4. Move into the self-organising-systems folder.
5. Run `pip install .`
6. Run `pip install jax==0.4.23 jaxlib==0.4.23`

## Optional: Install Weights and Biases

To see the progress of simulations and have a feature-rich analysis dashboard, Weights and Biases is recommended. Set it up as follows:
1. Create a Weights and Biases account and project [here](https://wandb.ai/site).
2. Run `wandb.login()` in the terminal.
3. Provide your API key when prompted (you can get this key in the settings of your Weights and Biases profile).
4. Set `USE_WANDB = False` in the starting point code.

**Note:** When not using Weights and Biases, set `USE_WANDB = False`. This is the default setting.

## Starting Point

To guide you through making the simulations, we have created a starting point Jupyter notebook. This notebook explains how to run the simulations without having to go through all of the scripts.

## Repository Structure

The following table explains each folder in the repository and its purpose.

| Folder             | Description                                                                                  |
|--------------------|----------------------------------------------------------------------------------------------|
| `utils`            | Contains all the necessary Python scripts to run, evaluate, and set up the experiments.      |
| `configs`          | Configurations used for simulations, specifically season-config for main simulations.        |
| `overrides`        | Overrides scripts in the base BiomakerCA package to add desired functionality.               |
| `notebooks`        | Notebooks used for trying out experiments.                          |
| `analysis_results` | Provides CSV file checkpoints of different simulations over the years for statistical tests. |
| `images`           | Storage for all images used in the project.                                                  |
| `videos`           | Contains simulation output videos.                                                          |


## Results

# Results

Here are the results of our simulations, showcasing the impact of sudden temperature increases in winter on plant biomass throughout the year.

<table>
  <tr>
    <td><img src="path/to/image1.png" alt="Image 1" width="300"/></td>
    <td><img src="path/to/image2.png" alt="Image 2" width="300"/></td>
    <td><img src="path/to/image3.png" alt="Image 3" width="300"/></td>
  </tr>
  <tr>
    <td><img src="path/to/image4.png" alt="Image 4" width="300"/></td>
    <td><img src="path/to/image5.png" alt="Image 5" width="300"/></td>
    <td><img src="path/to/image6.png" alt="Image 6" width="300"/></td>
  </tr>
  <tr>
    <td><img src="path/to/image7.png" alt="Image 7" width="300"/></td>
    <td><img src="path/to/image8.png" alt="Image 8" width="300"/></td>
    <td><img src="path/to/image9.png" alt="Image 9" width="300"/></td>
  </tr>
  <tr>
    <td><img src="path/to/image10.png" alt="Image 10" width="300"/></td>
    <td><img src="path/to/image11.png" alt="Image 11" width="300"/></td>
    <td><img src="path/to/image12.png" alt="Image 12" width="300"/></td>
  </tr>
  <tr>
    <td><img src="path/to/image13.png" alt="Image 13" width="300"/></td>
    <td><img src="path/to/image14.png" alt="Image 14" width="300"/></td>
    <td><img src="path/to/image15.png" alt="Image 15" width="300"/></td>
  </tr>
  <tr>
    <td><img src="path/to/image16.png" alt="Image 16" width="300"/></td>
    <td><img src="path/to/image17.png" alt="Image 17" width="300"/></td>
    <td><img src="path/to/image18.png" alt="Image 18" width="300"/></td>
  </tr>
  <tr>
    <td><img src="path/to/image19.png" alt="Image 19" width="300"/></td>
    <td><img src="path/to/image20.png" alt="Image 20" width="300"/></td>
     <td><img src="path/to/image20.png" alt="Image 20" width="300"/></td>
  </tr>
</table>

<p align="center">

<img width="400" height="300" src="https://github.com/JulianRodd/NaCO_project/blob/main/images/weights_and_biases/simulation.png?raw=true">

</p>



# Simulating the Effect of Sudden Temperature Increase in Winter on Plant Biomass - A BioMakerCA Project

This repository is made for the project described in "Simulating the effect of sudden temperature increase in winter on plant biomass" by Janneke Nouwen, Laura Stritzel, and Julian Roddeman.
This repository is based on the [self-organising-systems](https://github.com/google-research/self-organising-systems/tree/master) repository and see [this web article](https://google-research.github.io/self-organising-systems/2023/biomaker-ca/) or [this paper](https://arxiv.org/abs/2307.09320) for explanations of the base package.

## Introduction

Climate change is causing changes in ecosystems, leading to unusual plant behaviors like blooming in winter. This study focuses on the impact of rising winter temperatures on plant biomass.

The main research question is: **How do sudden warmer temperatures in winter influence plant biomass throughout the year?**

To answer this question, we created simulations using Python 3.11 and the Biomaker CA package, which employs Cellular Automata (CA) to model biomes. By simulating seasonal changes and introducing a sudden temperature increase in winter, we observe plant growth and reproduction to understand the impacts of an unusually warm winter. This repository includes all necessary scripts, configurations, and results for replicating our study.


![](https://github.com/JulianRodd/NaCO_project/blob/main/videos/spedup.gif?raw=true)

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

To guide you through making the simulations, we have created a starting point Jupyter notebook. This notebook explains how to run the simulations without having to go through all of the scripts. Click here to open the starting point: [Starting Point Jupyter Notebook](./starting_point.ipynb).

## Repository Structure

The following table explains each folder in the repository and its purpose.

| Folder             | Description                                                                                  |
|--------------------|----------------------------------------------------------------------------------------------|
| `utils`            | Contains all the necessary Python scripts to run, evaluate, and set up the experiments.      |
| `scripts`            | Scripts used to run experiments.     |
| `configs`          | Configurations used for simulations, specifically season-config for main simulations.        |
| `overrides`        | Overrides scripts in the base BiomakerCA package to add desired functionality.               |
| `notebooks`        | Notebooks used for trying out experiments.                          |
| `analysis_results` | Provides CSV file checkpoints of different simulations over the years for statistical tests. |
| `images`           | All images used in the project.                                                  |
| `videos`           | Contains simulation output videos.                                                          |


## Results
Here are the results of our simulations, showcasing the impact of sudden temperature increases in winter on plant biomass throughout the year.

<table>
  <tr>
    <td><img src="https://github.com/JulianRodd/NaCO_project/blob/main/images/weights_and_biases/Section-2-Panel-0-j9r1cpura.png?raw=true" alt="Section-2-Panel-0-j9r1cpura" width="300"/></td>
    <td><img src="https://github.com/JulianRodd/NaCO_project/blob/main/images/weights_and_biases/Section-2-Panel-0-lb8fyomrk.png?raw=true" alt="Section-2-Panel-0-lb8fyomrk" width="300"/></td>
    <td><img src="https://github.com/JulianRodd/NaCO_project/blob/main/images/weights_and_biases/Section-2-Panel-1-ojl7zqhlo.png?raw=true" alt="Section-2-Panel-1-ojl7zqhlo" width="300"/></td>
  </tr>
  <tr>
    <td><img src="https://github.com/JulianRodd/NaCO_project/blob/main/images/weights_and_biases/Section-2-Panel-1-sqi2jsjju.png?raw=true" alt="Section-2-Panel-1-sqi2jsjju" width="300"/></td>
    <td><img src="https://github.com/JulianRodd/NaCO_project/blob/main/images/weights_and_biases/Section-2-Panel-10-lz0y2d8ln%20(2).png?raw=true" alt="Section-2-Panel-10-lz0y2d8ln" width="300"/></td>
    <td><img src="https://github.com/JulianRodd/NaCO_project/blob/main/images/weights_and_biases/Section-2-Panel-11-zr9sd5on8.png?raw=true" alt="Section-2-Panel-11-zr9sd5on8" width="300"/></td>
  </tr>
  <tr>
    <td><img src="https://github.com/JulianRodd/NaCO_project/blob/main/images/weights_and_biases/Section-2-Panel-2-2jgx3uvuw.png?raw=true" alt="Section-2-Panel-2-2jgx3uvuw" width="300"/></td>
    <td><img src="https://github.com/JulianRodd/NaCO_project/blob/main/images/weights_and_biases/Section-2-Panel-2-744uunw6t.png?raw=true" alt="Section-2-Panel-2-744uunw6t" width="300"/></td>
    <td><img src="https://github.com/JulianRodd/NaCO_project/blob/main/images/weights_and_biases/Section-2-Panel-3-0ybf6n8s3.png?raw=true" alt="Section-2-Panel-3-0ybf6n8s3" width="300"/></td>
  </tr>
  <tr>
    <td><img src="https://github.com/JulianRodd/NaCO_project/blob/main/images/weights_and_biases/Section-2-Panel-3-wbs9psiys.png?raw=true" alt="Section-2-Panel-3-wbs9psiys" width="300"/></td>
    <td><img src="https://github.com/JulianRodd/NaCO_project/blob/main/images/weights_and_biases/Section-2-Panel-4-huz4wtr1a.png?raw=true" alt="Section-2-Panel-4-huz4wtr1a" width="300"/></td>
    <td><img src="https://github.com/JulianRodd/NaCO_project/blob/main/images/weights_and_biases/Section-2-Panel-4-qz4pfsynp.png?raw=true" alt="Section-2-Panel-4-qz4pfsynp" width="300"/></td>
  </tr>
  <tr>
    <td><img src="https://github.com/JulianRodd/NaCO_project/blob/main/images/weights_and_biases/Section-2-Panel-5-gpbbwc7k2.png?raw=true" alt="Section-2-Panel-5-gpbbwc7k2" width="300"/></td>
    <td><img src="https://github.com/JulianRodd/NaCO_project/blob/main/images/weights_and_biases/Section-2-Panel-5-tyyd90hhf.png?raw=true" alt="Section-2-Panel-5-tyyd90hhf" width="300"/></td>
    <td><img src="https://github.com/JulianRodd/NaCO_project/blob/main/images/weights_and_biases/Section-2-Panel-6-00b6vefb3.png?raw=true" alt="Section-2-Panel-6-00b6vefb3" width="300"/></td>
  </tr>
  <tr>
    <td><img src="https://github.com/JulianRodd/NaCO_project/blob/main/images/weights_and_biases/Section-2-Panel-6-o6fiexhau.png?raw=true" alt="Section-2-Panel-6-o6fiexhau" width="300"/></td>
    <td><img src="https://github.com/JulianRodd/NaCO_project/blob/main/images/weights_and_biases/Section-2-Panel-7-52xwhdhql.png?raw=true" alt="Section-2-Panel-7-52xwhdhql" width="300"/></td>
    <td><img src="https://github.com/JulianRodd/NaCO_project/blob/main/images/weights_and_biases/Section-2-Panel-7-ob5f4qybg.png?raw=true" alt="Section-2-Panel-7-ob5f4qybg" width="300"/></td>
  </tr>
  <tr>
    <td><img src="https://github.com/JulianRodd/NaCO_project/blob/main/images/weights_and_biases/Section-2-Panel-8-61ecpdsvc.png?raw=true" alt="Section-2-Panel-8-61ecpdsvc" width="300"/></td>
    <td><img src="https://github.com/JulianRodd/NaCO_project/blob/main/images/weights_and_biases/Section-2-Panel-8-8qpet55o6.png?raw=true" alt="Section-2-Panel-8-8qpet55o6" width="300"/></td>
    <td><img src="https://github.com/JulianRodd/NaCO_project/blob/main/images/weights_and_biases/Section-2-Panel-9-0c9ley12o.png?raw=true" alt="Section-2-Panel-9-0c9ley12o" width="300"/></td>
  </tr>
  <tr>
    <td><img src="https://github.com/JulianRodd/NaCO_project/blob/main/images/weights_and_biases/Section-2-Panel-9-kjqph0bfi.png?raw=true" alt="Section-2-Panel-9-kjqph0bfi" width="300"/></td>
    <td><img src="https://github.com/JulianRodd/NaCO_project/blob/main/images/weights_and_biases/air_soil_burnins.png?raw=true" alt="air_soil_burnins" width="300"/></td>
    <td><img src="https://github.com/JulianRodd/NaCO_project/blob/main/images/weights_and_biases/root-to-leaves.png?raw=true" alt="root-to-leaves" width="300"/></td>
  </tr>
</table>

# neurons_and_arts

<!-- This project implements a complete pipeline for extracting tetrahedral meshes of individual and grouped brain cell data and performing EMI simulations on those meshes. -->
Pipeline for automatic generation of single cell and dense brain tissue meshes from the [MICroNS](https://www.microns-explorer.org/cortical-mm3) dataset using [EMI-Meshing](https://github.com/scientificcomputing/EMI-Meshing).

![Cells](videos/all_15x15.gif)

## Overview

The workflow follows a sequence of steps from raw data configuration to physical simulation:

### 1. Configuration Generation
**Notebook:** `notebooks/generate_configs.ipynb`
- Define cell types (neurons, astrocytes, microglia, etc.), resolutions (MIP levels), and image processing parameters.
- Generate `.yml` configuration files that drive the EMI-Meshing pipeline.

### 2. Meshing (EMI-Meshing)
**Tool:** EMI-Meshing (https://github.com/scientificcomputing/emimesh)
- Process the generated configs to download raw EM data and generate tetrahedral meshes.
- Produce `.xdmf` mesh files.

### 3. Dataset Curation
**Notebook:** `notebooks/file_handling.ipynb`
- Prune failed meshes and remove redundant data files optimize storage.
- Sync datasets from a server to local environments and package them for distribution.

### 4. Parameter Analysis & Validation
**Notebook:** `notebooks/analyze_neuron_processing.ipynb`
- Systematically analyze the impact of resolution, morphological operations, and smoothing on mesh quality.
- Determine the optimal processing parameters for the final dataset.

### 5. Visualization
**Notebook:** `notebooks/visualize_cells.ipynb`
- Visualize the 3D meshes of reconstructed cells.
- Generate rotating GIFs of individual cells and cell cubes.

---

## Setup & Installation

### Basic Requirements:
- **Conda**: For environment management.
- **Snakemake**: For driving the EMI-Meshing pipeline.
- **FEniCSx**: For the EMI simulation.
- **PyVista & ImageIO**: For visualization and GIF generation.

### Clone github
```bash
git clone https://github.com/SamiLaubo/neurons_and_arts.git
cd neurons_and_arts
git clone https://github.com/SamiLaubo/emimesh.git
```

### Setup conda env
```bash
conda create -c conda-forge -c bioconda -n snakemake snakemake snakemake-storage-plugin-http snakemake-executor-plugin-cluster-generic -y
```
Install some extra packages for visualization
```bash
conda activate snakemake
conda install imageio ipywidgets ipykernel tqdm pip
pip install "pyvista[jupyter]" plyfile
```

## Usage

1. Generate configurations with notebooks/generate_configs.ipynb
2. Run EMI-Meshing pipeline to generate meshes
3. Go through file_handling.ipynb to remove unwanted data
4. Visualize results with visualize_cells.ipynb

### Running the EMI-Meshing Pipeline
To run a specific configuration:
```bash
snakemake --configfile config_files/neuron.yml --use-conda --cores 8
```

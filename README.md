
This repository contains relevant code written for the paper "A cell-based framework for modeling cardiac mechanics". It's a mechanical continuum model in which the cells are explicitly represented, dividing the domain into an extracellular space and an intracellular space. The code in this repo specifically considers virtual passive deformation and active contraction experiments, tracking important mechanical properties like displacement, strain and stress.

# Dependencies and Installation

The code was developed in Python, and is based on [FEniCS](https://fenicsproject.org/) (2019.1). Using conda, one can create an environment with all necessesary dependencies giving the command

`$ conda env create -f environment.yml`

You can install the software using e.g. pip in the root folder:

`$ pip install .`

With the software being installed, you should be able to run all scripts in the "scripts" folder.

# Scripts and demos

## Minimal working example demos

For exploring the core of the model, there are two simpler scripts in the "mwe\_demos" folder. These are standalone examples, which can be run without any dependencies to any other parts of the code. They do, however, require a Fenics installation.

Here you can find an active contraction example and a stretching example. Both use a coarse geometry, representing a single cardiac cell, and demonstrate how material properties are assigned – one value for the intracellular domain, one for the extracellular domain. These capture the core of the model itself, and might provide a good starting point if you want to understand the core of the implementation – e.g. for incoporate the EMI framework in your own code.

## Demos

This folder contains demos, simple examples which shows how the model can be used when imported as a library.

## Scripts

The scripts folder contains various scripts relevant for different kinds of experiments that can be performed by adjusting material parameters and deformation modes from the command line, or was performed for the paper explicitly. The scripts include

- simple\_demo.py: Small simple example of how one can use the code; simulates a contraction using a fairly coarse mesh, with no output saved.
- find\_optimal\_params\_experimental.py: Scheme for finding optimal material parameters in the EMI model based on experimental data; note that we do not provide the underlying data (however the data we used can be downloaded [from here](https://dataverse.tdl.org/dataverse/RVMechanics).
- active\_contraction.py: Extended code for contraction with options for choosing the mesh, varying material properties, with output optionally being saved as npy and xdmf files.
- passive\_deformation.py: Code for running the stretch or shear experiments, with options for choosing the deformation direction ("stretch_ff", "shear_fs", "shear_fn", "shear_sf", "stretch_ss", "shear_sn", "shear_nf", "shear_ns, or "stretch_nn"), the mesh, varying material properties, with output optionally being saved as npy and xdmf files.
- sa\_step1.py, sa\_step2.py, sa\_step3.py: The three-phase scripts used for Sobol analysis. Here, it is beneficial to split them in three phases where the first generates the parameter space, the second performs the individual experiments (~15 min running time per each, for each parameter combination and each deformation mode), then the third one calculates the Sobol indices.

The script parameter\_setup.py is not used directly, but is used to add command line arguments to most of the above scripts.

If applicable, the xdmf files (and the corresponding h5) files can be opened in Paraview. The npy files contains averaged load, strain and stress values, and can easily be read to a Python dictionary using

```
data = np.load(filename, allow_pickle=True).item()
```

# Meshes

Meshes for various cell configurations are avaible in the "meshes" folder. The meshes are all generated using [tieler](https://github.com/MiroK/tieler/), with geometries as given in tile.geo (same for all experiments). A mesh typically have a name like

```
tile_connected_5p0.h5 
```

or 

```
tile_connected_5p0_1_2_2.h5
```

where the first part gives the mesh resolution (20p0 = 20.0, 10p0 = 10.0, etc; numbers corresponding to Gmsh's characteristic length), and the 3 last numbers denotes the number of cardiac cells; so "1\_2\_2" means we have a multicellular domain consisiting of 1 x 2 x 2 cells.

There are also a number of 2D meshes (look for the prefix "2D").

# Publications
The code in this repo has been used for a few publications. The code is always evolving and results as reported in one paper might potentially not be reproducable using the most recent version of the code (although I try to minimize the differences). There are frozen versions at Zenodo which might be downloaded to get the code as it was:
- Åshild Telle, Joakim Sundnes, Samuel T. Wall: *Modeling Cardiac Mechanics on a Sub-Cellular Scale*. *Modeling Excitable Tissue*, 2021: [Publication](https://link.springer.com/chapter/10.1007/978-3-030-61157-6_3) - [code (Zenodo)](https://zenodo.org/records/3769029)
- Åshild Telle, James D. Trotter, Xing Cai, Henrik Finsberg, Miroslav Kuchta, Joakim Sundnes & Samuel T. Wall: *A cell-based framework for modeling cardiac mechanics*. *Biomech Model Mechanobiol*, 2023: [Publication](https://link.springer.com/article/10.1007/s10237-022-01660-8) - [code (Zenodo)](https://zenodo.org/records/7137689)

# Contributors, issues, questions

This software is developed and maintained by Åshild Telle (Simula Reserach Laboratory; University of Washington). Any issues, questions or general requests can be sent to aashild@uw.edu.


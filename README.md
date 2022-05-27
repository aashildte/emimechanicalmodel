
This repository contains relevant code written for the paper "A cell-based framework for modeling cardiac mechanics". It's a mechanical continuum model in which the cells are explicitly represented, dividing the domain into an extracellular space and an intracellular space. The code in this repo specifically considers virtual passive deformation and active contraction experiments, tracking important mechanical properties like displacement, strain and stress.

# Dependencies and Installation

The code was developed in Python, and is based on [FEniCS](https://fenicsproject.org/) (2019.1). Using conda, one can create an environment with all necessesary dependencies giving the command

`$ conda env create -f environment.yml`

You can install the software using e.g. pip in the root folder:

`$ pip install .`

With the software being installed, you should be able to run all scripts in the "scripts" folder.

# Minimal working example demos

For exploring the core of the model, there are two simpler scripts in the "mwe\_demos" folder. These are standalone examples, which can be run without any dependencies to any other parts of the code. They do, however, require a Fenics installation.

Here you can find an active contraction example and a stretching example. Both use a coarse geometry, representing a single cardiac cell, and demonstrate how material properties are assigned – one value for the intracellular domain, one for the extracellular domain. These capture the core of the model itself, and might provide a good starting point if you want to understand the core of the implementation – e.g. for incoporate the EMI framework in your own code.

# Scripts

The scripts folder contains various scripts relevant for different kinds of experiments; some easy, some more advanced, some general, some specialzied for certain experiments. The scripts include

- simple\_demo.py: Small simple example of how one can use the code; simulates a contraction using a fairly coarse mesh, with no output saved.
- find\_optimal\_params\_experimental.py: Scheme for finding optimal material parameters in the EMI model based on experimental data; note that we do not provide the underlying data (however the data we used can be downloaded [https://dataverse.tdl.org/dataverse/RVMechanics from this repository]).
- active\_contraction.py: Extended code for contraction with options for choosing the mesh, varying material properties, with output optionally being saved as npy and xdmf files.
- passive\_deformation.py: Code for running the stretching or shearing experiments, with options for choosing the deformation direction, the mesh, varying material properties, with output optionally being saved as npy and xdmf files.

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

# Contributors, issues, questions

This software is developed and maintained by Åshild Telle, as an employee at Simula Research Laboratory. Any issues, questions or general requests can be sent to aashild@simula.no.

To the best of our knowledge, one should be able to reproduce all experiments performed in the paper using this code. However, we have also tried to make the code nice and readable before publishing it – and with any code change there is a risk of results being altered if produced again as well.

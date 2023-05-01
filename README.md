# ConnecMap
A python toolbox to derive connectivity-based brain parcellation from a connectivity data.

# Installation
Python 3.6 or later is required.
## Optional dependencies
If you want to use this package to parcellate mouse isocortex based on the AIBS voxelized connectome, the following dependencies are required to access that data. If you want to use a different data source, related to a different species, they are not required (but you will most likely have to implement access to that source yourself, see below).
### (Optional) Install mouse_connectivity_models

Install the package _allensdk_ available on pypi:
> pip install allensdk

Install the package mouse-connectivity-models available on the github pages of the AIBS:
> https://github.com/AllenInstitute/mouse_connectivity_models.git 

## Installation of packages
Install our utility package for accessing connectome data and diffusion embedding:
> cd diffmap_utility  
> pip install .

Install our utility package for reversal detection and configuration management:
> cd custom_parcellations  
> pip install .

# Overview
In this description, I assume that the reader is familiar with the manuscript describing the methods and analyses used. If you are not familiar, go read it at: https://www.biorxiv.org/content/10.1101/2022.08.30.505842v1.abstract 

## Structure
The code in this reqpositort is split into two separate projects: 

_diffmap_utility_ is used to access data sources of brain connectivity to get voxel-to-voxel connection strength data, then perform the diffusion embedding process on it and store the result in a .nrrd file containing the value of each component associated with each voxel. Plus a .json file containing the relative strength (variance explained) of each component.

_custom_parcellations_ is used to detect reversals in that data and derive a parcellation from it. It also performs analyses of the parcellation and overall management of a parcellation project, i.e. reading the scientific and technical configuration and preparing the next diffusion embedding step.

You will have to define and configure a source of brain connectivity data. Currently, only two types of sources are implemented: One, based on a propriatory, .h5-based file format and one accessing the voxelized mouse brain connectome of the Allen Institute. Using other sources will require some coding (see below). If you deicde to do so, please contribute your implementation to this repository.

# Running a parcellation project
## Basic steps
Start by acquiring the required files and configurations (see below). As a starter, refer to the example contained in this repository. Place the files into a project directory. Note that the locations of input files relative to the project directory must be exactly specified in the configuration file (see below and the example).  
Then:  
Open python and run:
> from parcellation_project import ParcellationProject  
p = ParcellationProject("/path/to/project/directory", resume=False)  
p.flatten_current_level(components_to_use=list(range(100)), overwrite=True)

This results in configuration files for the diffusion embedding process being written to the project directory. The terminal output of the function call will provide the user with the exact command to run to execute the process. The reason we provide the command instead of directly executing the process is as follows: The process is **very** memory and cpu-intensive. Details depend on the size of the connectome, but you probably want to run this on an allocation of a dedicated computing system. This would be hard to automate for all conceivable systems, so we instead provide the command to run and leave this to the user.  

After the process has be successfully run, resume the project the following way and use the output to split the region of interest into subregions:  
> from parcellation_project import ParcellationProject  
p = ParcellationProject("/path/to/project/directory", resume=True)  
p.split_current_level()  

This will create a new region annotation file and a new hierarchy file in the project directory that define the split of the root region of interest into subregions. You can then apply analyses that were configured in the main configuration file, but **this part is of course optional**:
> p.analyze_current_flatmap()  # Analyzes the diffusion embedding output
p.analyze_current_parcellation()  # Analyze the parcellation result  

For details on the types of analyses, refer to the manuscript. If you decide that further, more fine-grained parcellation is needed, you can split the subregions further by running the following command again:
> p.split_current_level()  

This will again provide the command to run the diffusion embedding process and will be followed logically by:
> ...  
p.split_current_level()  

as above. Continue this, until you decide that further parcellation is not required.

## Required files
A number of files is required to run a parcellation project. Some of them are data source-independent, i.e. required for any project, some are only required for a given source of brain connectivity data.  
**Note that this repository contains examples of all these files.**

### Source-independent files
- **Voxel annotations**: This is a .nrrd file that associates each voxel of the brain schema used with an integer that denotes a brain region. It is used to find the basic brain region (such as Isocortex) the user wants to parcellate. One output of the entire process will be a modified version of this file with new annotations for that region that define the new parcellation.  
**Note: This file defines the shape and resolution of the brain schema used. All other .nrrd files in the project must have the same shape and resolution!**  
One more note about resolution: This package has only been tested for resolutions of 100 um. In principle, other resolutions should work, but we cannot guarantee that at this point.
- **Hierarchy file**: This is a .json file that translates that translates the integers used in the voxel annotations to named brain regions. It also defines the hierarchy of brain regions. This allows us to look up which atomic brain regions are contained within the larger region the user wants to parcellate. Another output of the entire process is a modified version of this file where the hierarchy under that larger region is replaced with the newly derived one.  
The format of this file is the same used by the AIBS CCF to define the mouse brain hierarchy. It is very intuitive, just check the example hierarchy contained in this repository. Alternatively, seek out the documentation on the AIBS pages.  
- **Antomical flatmap**: The entire process is based on reversals of two-dimensional gradients. But a brain is three-dimensional. A flatmap defines the projection down to two dimensions. This is very cortex-centric, where one is typically interested in a parcellation orthogonal to the cortical layers, hence one would project columns spanning all layers to the same two-dimensional pixel.  
This is a .nrrd file that associates each voxel with two integers that specify its flat x- and y- location. Voxels that have the same x- and y- locations will be treated together, i.e. they will always end up in the same brain region. In the project our manuscript is based on, an average of 13 +- 8 voxels were mapped to the same location. We recommend using similar numbers though this is not strictly required.  
**Note: The anatomical flatmap must not have holes**: That is, there must not be regions in the flat coordinate sytem with zero voxels mapped to them that are completely surrounded by regions with >0 voxels.  
- **The main configuration file**: This file configures which data source to use, the locations of other input files, which parcellation algorithms to use and their paramters, and which analyses to perform. For the exact format, see below.

### Source-dependent files
When using the AIBS voxelized mouse brain connectome as a data source, the following files are also required.
- **structures file**: This is a .json file that also contains the integer ids associated with each region and the region hierarchy. It is largely redundant, but used by mouse_connectivity_models. In the future we may implement code that creates this file from the already required hierarchy file, but for now it is provided by the user.
- **voxel_model_manifest**: This is a .json file that is used as the data configuration for mouse_connectivity_models. It specifies where to download the files containing the connectivity data. 

## Configuration file format
This is a .json file that configures the entire process, both technically and scientifically. Just like the required files, some entries are source-independent, and some are source-dependent.  
### Source-independent entries
- **level_class**: Specifies which python class to use to access the brain connectivity data. Different classes are to be used for different data sources. The class must exist as an class in __parcellation_project.parcellation_levels__. Which source-depdendent entries are required depends on the value of this entry.  
- **parameters**: Specifies the scientific paramters of the parcellation process.
  - **splitting**: Parameterized exactly how to perform the splitting into subregions.
    - **step_1**: How to perform the initial (raw) split.
      - **function**: Which function to use for the initial split. Must exist under __parcellation_project.split__.
      - **args**: List of additional arguments to give to the function specified above. Details depend on the function, check that functions documentation.
      - **kwargs**: List of additional keyword arguments to give to the function specified above. Details depend on the function, check that functions documentation.
    - **step_2**: How to post-process and refine the split.
      - **function**: Which function to use to post-process the initial split. Must exist under __parcellation_project.split__.
      - **args**: List of additional arguments to give to the function specified above. Details depend on the function, check that functions documentation.
      - **kwargs**: List of additional keyword arguments to give to the function specified above. Details depend on the function, check that functions documentation
  - **visualization**: Which visualizations of the resulting split to generate. Each must exist as a function under __parcellation_project.analyses__.  
    - **step_1**: For the output of the raw split.
    - **step_2**: For the output of the post-processed split.
  - **initial_parcellation**: Which parcellation to start with. Must be a dict with the desired names as keys and lists of brain regions contained in each as values. The names can be anything, but the brain regions in the values must exist in the hierarchy file used (see above).  
  You will most probably want to simply put a single entry here with a generic name that contains all brain regions of interest; then the structure underneath is determined by this tool box. But if you want to start with a certain pre-determined structure, this is where you would put it.  
  - **Diffusion_mapping**: Parameters related to running the diffusion embedding process.  
    - **consider_connectivity**: Which connectivity should be used as the basis of the diffusion embedding process. Currently, only a single case is implemented: **inter**. This considers the connectivity of each voxel with the entirety of the regions to parcellate. In the future, we want to implement a use case where the connectivity with a __different__ region is considered instead, e.g. parcellation of cortex based on its thalamic connectivity.
    - **connectivity_direction**: After the above entry specifies which regions should be considered for the connectivity of a voxel, this specifies the direction to use. Possible values are: __afferent__, connectivity from those regions to a voxel, __efferent__, connectivity from the voxel to those regions, or __both__, the concatenation of both connectivities.
  - **level_configuration**: Mostly parameterizing the locations of input files, and the analyses to run on the resulting parcellations.
    - **root_region**: The name of the brain region to parcellate. Note that parameters/initial_parcellation **must** define a parcellation of this region!
    - **hemisphere**: Which hemisphere to parcellate. At this point, parcellations for hemispheres must be generated independently.
    - **inputs**: Where to find the required input files. Key specifies type of the file, values the path to the file. All non-absolute paths are interpreted relative to the location of this configuration file. For a list of required files, see above.
    - **paths**: Where to put files that are generated during the process of running the parcellation project. Note that this tool box generates a parcellation by recursively splitting the root region of interest into smaller and smaller regions. This is reflected in the file structure of the output, where each successive split adds another level to it.  
    The root of all outputs is the location of this configuration file.
      - **lower_level**: This directory is created at the root to hold the files required to run the diffusion embedding and the output/results/analyses. Subsequent runs of additional splits will recursively create another directory with this name underneath that. This leads to the hierarchy of splits being reflected by the file system structure.  
      All paths defined by the following entries are relative to this root associated with the "level" of a split.
      - **region_volume**: Where to put the .nrrd file of resulting region annotations. This is the main output, as the annotations define a parcellation. 
      - **hierarchy**: Where to put the .json file defining the resulting region hierarchy.
      - **flatten_cfg**: Where to put the file defining the scientific configuration of the diffusion embedding process.
      - **cache_cfg**: Where to put the file defining the technical configuration of the diffusion embedding process.
      - **flatmap**: Where to put the results of the diffusion embedding process.
      - **characterization**: Where to put more results of the diffusion embedding process (fraction of variance explained by each component).
      - **analyses**: Directory to put the results of analyses into (such as plots).
    Additionally, when **AibsLevel** is used as the "level_class":
      - **structures**: The location to put modified "structures" files
      - **manifest**: The location to put modified "voxel_model_manifest" files.

# Customization
As evidenced by the structure of the configuration file, this tool box is rather modular and is intended to provide different alternatives for each step of the process. Thus, you might want to implement your own. This could be custom analyses to run on the resulting parcellation, or alternative reversal detector functions. We encourage this of course and only ask you to consider contributing your work back to our repository with a pull request.  

But, probably the most important type of customization is to implement a new data source for voxelized brain connectivity. Here, we will outline the steps required to do this.  

## Adding a data source
The main work will be implementing access to the data source in the __diffmap_utility__ package. But also some additions to __custom_parcellation__ will be needed.

### ... To diffmap_utility
A new data source is implemented as a class derived from __projection_voxels.CachedProjections__ with functions implemented that configure access to the data. We will give a vague outline here, for a more detailed understanding, refer to the source code.  
1. Think of a way to implement a __voxel_array__ for your data source.  
Base access to connectivity data is performed using a data structure that supports a 2d indexing with __source__ voxels of connectivity along the first, and __target__ voxels of connectivity along the second dimension. That is, it behaves like a two-dimensional numpy.array. It does not have to be one, just support similar indexing. Additionally, two numpy.arrays will need to be provided, that specify the 3d coordinates of the __source__ and __target__ voxel respectively. 
2. Derive a class from __CachedProjections__ with a constructor specific to your data source. The constructor can do any complex task, but it has to eventually call the constructor of the super-class, providing the three data structures described above.
3. Register your class in __diffmap_utility/projection_voxels/projection_from_config.py__. In that file, implement the code required to instantiate your class from specifications in a .json configuration file. Structure of that file is:
> {  
    "class": ("Name of your class"),  
    "args": {(dict of any kwargs your need)}  
}  

### ... To custom_parcellations
Certain changes are required to the __custom_parcellations__ part as well to allow the tool box to correctly parameterize the invokation of the diffusion process. For an example, check __custom_parcellations/parcellation_project/parcellation_levels/aibs_level.py__ that implements them.  

1. Implement a class derived from __parcellation_project.parcellation_levels.ParcellationLevel__ and place it under __parcellation_project.parcellation_levels__. For details refer to the source code. This class has to implement the following functions:  

  (a) __find_inputs_from_config__: A classmethod, used to initialize a project from the main configuration file. The function must read and return all inputs required by the constructor. The superclass version of this (ParcellationLevel.find_inputs_from_config) can be used to get the brain region hierarchy (as a voxcell.RegionMap) and the region annotations (as a 3d numpy.array) of the brain studied.  
  This function also truncates the brain region hierarchy of the region to parcellate to "make room" for the new parcellation. 

  (b) __find_inputs_from_file_system__: A classmethod used to resume an existing parcellation project, for example after an iteration of the diffusion embedding process has been run to completion. It also must read and return all inputs required by the constructor. But instead of relying on the input files (and for example truncating the hierarchy defined therein) it insteads reads and returns existing, previously generated files in the project directory.

  (c) __the constructor__: The constructor of the base class requires the path to the project root, the contents of the configuration file, the region hierarchy (voxcell.RegionMap) and the region annotations (voxcell.VoxelData). But you might require additional inputs for your specific case, such as the url to access the connectivity data or an api token.

  (d) __cache_config__: Returns a dict that will be used as the input into the __projection_from_config__ function that is used to instantiate the class accessing the connectivity data when the diffusion process is run. For details, see above.

  (e) Any additional functionality that may be required.


2. (Optional) If there are additional inputs required for your class (consumed by __find_inputs_from_config__), add their path to the config file.  
3. Change the value of **level_class** in the config file to the name of your custom class (see above).

## Other contributions
Some other useful functionality is still not fully implemented and we invite you to implement it and contribute:
1. The ability to specify other value for **consider_connectivity** in the config file (see above)
2. Better handling of hemispheres. That one's a mess right now.
3. A command line script that performs the steps related to __custom_parcellations__ that are currently run in an interactive session. This may include automatically submitting the diffusion embedding as a computing job to a job scheduler, such as SLURM.

## Acknowledgement
This study was supported by funding to the Blue Brain Project, a research center of the École polytechnique fédérale de Lausanne (EPFL), from the Swiss government’s ETH Board of the Swiss Federal Institutes of Technology.

Copyright (c) Blue Brain Project/EPFL

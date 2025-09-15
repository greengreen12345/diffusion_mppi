# diffusion_mppi

## Installation
First, clone the repo in your folder and create the conda environment. 
````bash
cd <project_folder>
git clone https://github.com/tud-amr/m3p2i-aip.git

conda create -n m3p2i-aip python=3.8
conda activate m3p2i-aip
````

This project requires the source code of IsaacGym. Check for the [prerequisites and troubleshooting](https://github.com/tud-amr/m3p2i-aip/blob/master/thirdparty/README.md). Download it from https://developer.nvidia.com/isaac-gym, unzip and paste it in the `thirdparty` folder. Move to IsaacGym and install the package.
````bash
cd <project_folder>/m3p2i-aip/thirdparty/IsaacGym_Preview_4_Package/isaacgym/python
pip install -e. 
````

Then install the current package by:
````bash
cd <project_folder>/m3p2i-aip
pip install -e. 
````


## Run the scripts

There are two instances of Isaac Gym, one for throwing the rollouts and deriving the optimal solution, and one for updating the "real system". Please run the commands below in two terminals with activated python environment.

### Train the model
Run this terminal first:
````bash
conda activate m3p2i-aip
python3 reactive_tamp.py -cn config_panda multi_modal=True cube_on_shelf=True
````

Then run the second terminal:
````bash
conda activate m3p2i-aip
python train.py
````
### Test the model
Run this terminal first:
````bash
conda activate m3p2i-aip
python3 reactive_tamp.py -cn config_panda multi_modal=True cube_on_shelf=True
````

Then run the second terminal:
````bash
conda activate m3p2i-aip
python test_model.py
````


## **Instructions for Running Splatter Image: Ultra-Fast Single-View 3D Reconstruction Research Paper**

### Objective

This document aims to be a complete and thorough overview of how to get the code in the [research paper](https://szymanowiczs.github.io/splatter-image) running. Specifically, it aims to enable the reader to easily (by easily, we mean it should be straightforward to run after following some tedious steps below, hopefully without any additional Google searches) and successfully run a web-based application locally, that loads the pre-trained model hosted by the paper’s authors and perform inferences on custom inputs. More specifically, we are referring to the `gradio_app.py` file in the code [repository](https://github.com/szymanowiczs/splatter-image). Furthermore, following the steps below should also enable the reader to be able to perform evaluations on downloaded datasets using the `eval.py` script also in the [repository](https://github.com/szymanowiczs/splatter-image), which will be discussed in some more detail below.

### Assumptions

This guide assumes the user is inside a Linux environment. With Anaconda set up, it can be followed closely for Windows as well, however, instructions pertaining to environment variables and CUDA will differ. This guide does not require `sudo` system privileges, making it applicable to anyone with access to a Linux GPU environment. The guide also assumes having access to CUDA-enabled GPUs. If the reader intends to run the demo on CPU only, some additional changes to the `gradio_app.py` script will be required, which is not covered in this guide.

### Conda Environment Setup

There are usually two routes to take here. Since the project code base comprises primarily python source code with no pre-compiled binaries, it is possible to try running the code with the bleeding-edge versions of python, PyTorch, CUDA, etc. The other route to take is the one that is tried and tested, with the major downside being having to settle with older versions of libraries that are compute-intensive, meaning missing out on possible optimizations and other improvements. Still, for the sake of user-friendliness, the latter is recommended. Once the code is up and running in an environment with older library versions, it can then be possible to try with the bleeding-edge or newer versions of the same. This guide will try and support both routes.

Regardless of the route taken, start with creating a conda environment:

`conda create -n splatter-image`

And activate the same with the following command:

`conda activate splatter-image`

Feel free to rename the environment any way that is preferred.

#### Route 1: Bleeding-edge (not recommended)

In this new conda environment, start with installing Python, like follows:

`conda install python`

Or 

`conda install python=3.13`

Where the above can be substituted with any newer version 3.X. 

Next, install the latest version of PyTorch, by following the instructions on the [official website](https://pytorch.org/get-started/locally/):  
[https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)  
While doing so, please ensure the version of PyTorch is compatible with the downloaded Python version from earlier, and supported CUDA version by the NVIDIA driver as can be determined by running `nvidia-smi`. **Ensure the selected CUDA version is at most the version determined by `nvidia-smi`. It may be older than the displayed version, but not newer.** This guide assumes a setup with NVIDIA accelerator(s), with the drivers already configured and installed correctly. The steps followed to install PyTorch may differ in any other case (such as AMD or Apple Silicon). In those cases, please follow the installation instructions on the [official website](https://pytorch.org/get-started/locally/). 

Once PyTorch has been successfully installed, install the remaining project requirements by cloning the [repository](https://github.com/szymanowiczs/splatter-image) locally if you have not already done so, and then navigating to the repository locally while within the same conda environment and running:

`pip install -r requirements.txt`

This step may take a while; feel free to go grab a coffee ☕. Once this step is done, install the last few requirements:

`pip install rembg onnxruntime`

Alternatively, if you cloned our [fork](https://github.com/rhit-biswalt/splatter-image) of the repository, the above step may be skipped, since the `requirements.txt` file has been updated to include the above dependencies of the frontend demo app.

Regardless of how the dependencies above were installed, if using Python version \>= 3.13, please also install the following dependency (not in our fork’s `requirements.txt` ):

`pip install audioop-lts`

Next up are the instructions to install the appropriate version of the CUDA development toolkit, which is required to build the `diff-gaussian-rasterization` dependency as detailed below. To perform this step correctly, please identify the required version of cuda-toolkit that must be installed, by matching the cuda version determined by `nvidia-smi` from a previous step, and more importantly, matching the version of cuda that PyTorch was installed for. If unsure, please run the following:

`>>> import torch`  
`>>> torch.version.cuda`  
`'12.8' # this is the version you need, for example. May differ`

 At this point, it is recommended to skip to the next paragraph in this document, and it may be worth understanding why: This guide details the steps required to install the CUDA toolkit inside the conda environment that was created earlier, which is recommended to maintain a version compatible with other libraries previously installed in the environment. If choosing to take the **non-recommended** route of system-wide installation of the CUDA toolkit, please refer to [this](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/) installation guide (or [this](https://wiki.archlinux.org/title/GPGPU) for Arch Linux). After installing CUDA toolkit, use your system’s package manager to install `glm` (eg: `sudo pacman -S glm`  in Arch Linux). Additionally, if choosing to follow the system-wide installation route, please ensure all components of the appropriate CUDA toolkit version are properly installed in the system, and that the following environment variables are pointing to locations that look like below:  
`# Add these lines to your ~/.bashrc or ~/.profile file`   
`export PATH=/usr/local/cuda/bin:$PATH`   
`export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH`   
`export CUDA_HOME=/usr/local/cuda`

Assuming you are taking the **recommended** route of installing the CUDA toolkit in your conda environment, simply run the following, replacing version `12.8.0` with the appropriate CUDA version as determined earlier:  
`conda install -c nvidia/label/cuda-12.8.0 -c nvidia cuda-toolkit`  
If this step does not find the cuda-toolkit package, please ensure the CUDA version is listed [here](https://anaconda.org/nvidia/cuda-toolkit). If not, pick another CUDA version and reinstall PyTorch for this new version, and make sure the new CUDA version does not exceed the one indicated by `nvidia-smi.`

Next, export the following environment variables within the conda environment:

`export CUDA_HOME=$CONDA_PREFIX`  
`export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:LD_LIBRARY_PATH`

We will also need `glm`, a final dependency for building `diff-gaussian-rasterization`. In conda, run the following:

`conda install -c conda-forge glm`

Finally, the last step is to clone the [repository](https://github.com/graphdeco-inria/diff-gaussian-rasterization) at [https://github.com/graphdeco-inria/diff-gaussian-rasterization](https://github.com/graphdeco-inria/diff-gaussian-rasterization). Navigate to the parent folder where the repository was cloned (for example in my system i must perform a `cd ~/repos` since the repository is located at `~/repos/diff-gaussian-rasterization`). Then run:

`pip install diff-gaussian-rasterization/`

###### *No module named `torch` error*

In case of getting the above error from attempting to install `diff-gaussian-rasterization`, try `pip install --no-build-isolation diff-gaussian-rasterization/`  
This forces pip to use the version of Python installed in your conda environment, with all the appropriate dependencies installed earlier.

And that’s it\! Now try running the following (optionally modify the `--cuda` argument):

`python gradio_app.py --cuda 0`

##### Potential errors (official repository)

###### *No CUDA GPUs available*

If there is an error such as “no CUDA GPUs available” in the version of `gradio_app.py` from the [original repository](https://github.com/szymanowiczs/splatter-image), take a look at line 33 (as of May 23, 2025, if changed, try finding the below line), which may look like the following:

`os.environ["CUDA_VISIBLE_DEVICES"] = "1"`

Make sure this variable points to the appropriate GPUs in your system. For example, running it locally on a system with one GPU would require the following change to:

`os.environ["CUDA_VISIBLE_DEVICES"] = "0"`

After the above change, try running it again.

###### *pickle.UnpicklingError*

In the case of an error with loading the model, navigate to line 48, or a line that looks like the following:

`ckpt_loaded = torch.load(model_path, map_location=device)`

Then, add the keyword argument `weights_only=False` to the `torch.load()` to get the following:   
`ckpt_loaded = torch.load(model_path, map_location=device, weights_only=False)`

And run again.

#### Route 2: Older versions that have been tested (Recommended)

If taking the recommended route of configuring an environment with older versions of libraries that have been successfully tested in the past, follow the same instructions in \[Route 1\], but with one of the following combinations of library versions:

##### CUDA 12.8

#### CUDA version: 12.8

Python version: 3.13.2  
PyTorch version: 2.7.0+cu128

##### CUDA 11.7

CUDA version: 11.7  
Python version: 3.10.17  
PyTorch version: 1.13.1  
mkl version: 2024.0.0 (might need to downgrade mkl to 2024.0.0 for old versions of PyTorch with `conda install mkl=2024.0.0` )

### SSH Port-Forwarding to Run the Demo

If running the `gradio_app.py` demo via `ssh` into a server such as `hinton` or `gebru`, please connect one terminal to the server with the following command for port forwarding, replacing `hinton` with the appropriate ssh server (in `~/.ssh/config`) or username@server:  
`ssh -L 7860:localhost:7860 hinton`

Then, in a browser window, visit [http://localhost:7860](http://localhost:7860) 

### Dataset Downloading and Setup

Todo. mainly comment on PyTorch3D version for Co3D data preprocessing. Rest should be self-explanatory from the repo README. 

## 


# FAQ

We list some of potential problems encountered by us and users here, along with some solutions. We welcome users to enrich it by opening issues.

## Environment
* The general structure of this repository and the provided commands are carefully organized. Changing it (e.g. run `python train_generation.py --cfg ...` under `model/ugg`) would make the system hard to find the needed files.

* Installation of `xformers` might fail with conda environment file. If this happens, try install it from source:

  ```
  pip install -v -U git+https://github.com/facebookresearch/xformers.git@v0.0.21#egg=xformers
  ```

* We are still trying to compose a docker file suitable for both training, inference, and simulation. We are experiencing a conflict between isaacgym, xformers, and existing NVIDIA Optimized Frameworks. 

## Data Preparation

* Computing the LION's latent takes more time than we expected and makes the training less efficient. We recommend to presave a set of latents of (normalized) point clouds beofre training.

* Because of presaving, we assume the point clouds from dataset is already normalized if needed.

## Training

* To make full use of the LION model, a normalization factor **6.6** is used for objects suitable for grasping.

* In order to train the model within a tolerable time period, we used `xformers`, gradient checkpointing, and `bf16` for training. You will need `xformers` to load our model weights.


## Inferance

* We include three sets of hand parameters for `hand2obj` task. You may consult the dataset for more suitable hand parameters.

* For simplicity, we only support one GPU inference. For each iteration, one object and multiple scales are used for generation.


## Evaluation

* You might need to use docker for IsaacGym if you don't have sudo.

* Import IsaacGym at the beginning of your main file as we did in `isaac/simulation_test.py` if you want to write your own script. Importing after numpy/torch/etc. might cause errors.

* We find that IsaacGym fails when each simulation includes too many hand poses (>500?) or after ~30 times of `reset_simulator`. Therefore, large batch size should be splitted, and we write a sample script to restart evaluation every certain amounts of objects. The setting should be subjective to your evaulation size.

* On our machine, to run IsaacGym with GPU, a specific GPU should be assigned and it changes after certain amount of time. As we detailed in `isaac/simulation_test.py`, remember setting this magic number properly.

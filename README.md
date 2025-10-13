# Extreme Parkour with Legged Robots (Modified Fork)

<p align="center">
<img src="./images/teaser.jpeg" width="80%"/>
</p>

>  **This repository is a modified fork of [chengxuxin/etreme-parkour](https://github.com/chengxuxin/extreme-parkour).**  
> The goal of this fork is to improve clarity, compatibility, and usability for research and experimentation.

---

## Modifications in This Fork

This fork introduces several improvements and updates:
- Updated installation instructions for **CUDA 12 / RTX 40xx GPUs**
- Added Robot **unitree go2** for training
- Clarified usage examples for training and playing policies
- Improved documentation readability and structure

---

## Installation

```bash
# Create a new conda environment
conda create -n parkour python=3.8
conda activate parkour

# Install PyTorch depending on your GPU type

# For older GPUs (e.g., NVIDIA 30xx series)
pip3 install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html

# For newer GPUs (CUDA 12.1 / 40xx series)
pip3 install torch torchvision torchaudio -f https://download.pytorch.org/whl/cu121

# Clone this repository
git clone git@github.com:xxxxxyang/QRC25-parkour.git parkour
cd parkour

# Download Isaac Gym binaries from NVIDIA Developer:
# https://developer.nvidia.com/isaac-gym
# Originally trained with Preview 3, but Preview 4 also works fine.

# Install Isaac Gym
cd isaacgym/python && pip install -e .

# Install local packages
cd ~/parkour/rsl_rl && pip install -e .
cd ~/parkour/legged_gym && pip install -e .

# Install other dependencies
pip install "numpy<1.24" pydelatin wandb tqdm opencv-python ipdb pyfqmr flask scikit-learn
```

<!-- tips: If you find error like `libGL.so.1: cannot open shared object file`, try:
```bash
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGL.so.1
``` -->

---

## Usage

Change directory to scripts:

```bash
cd legged_gym/scripts
```

### 1. Train a base policy

```bash
python train.py --exptid <exptid_name> --device cuda:0 --task go2 --headless
```

* Recommended: 10–15k iterations (~8–10 hours on RTX 3090)
* If `--exptid` is not provided, a default one will be generated (e.g., `Oct-09_19_31-go2`).
* Available tasks: `a1`, `go2`, `climb_go2`, `leap_go2`

### 2. Train a distillation policy

```bash
python train.py --exptid <distill_exptid_name> --device cuda:0 --resume --resumeid <your_base_exptid> --delay --use_camera --headless
```

* Recommended: 5–10k iterations (~5–10 hours on RTX 3090)
* You can use any available GPU by setting `--device cuda:#`
* Default name: `Oct-10_22_46-go2-distill`

### 3. Play base policy

```bash
python play.py --exptid <your_base_exptid_name>
```

If you trained beyond 8k iterations, you may need to add `--delay`.

### 4. Play distillation policy

```bash
python play.py --exptid <your_distill_exptid_name> --delay --use_camera
```


### 5. Export trained models for deployment

```bash
python save_jit.py --exptid <your_exptid_name>
```

Traced models will be saved in:

```bash
legged_gym/logs/<proj_name>/<exptid>/traced/
```

---

## Viewer Controls

Can be used in both Isaac Gym and the web viewer.

| Action               | Key                       |
| -------------------- | ------------------------- |
| Move camera          | `ALT + Left Mouse + Drag` |
| Switch robot         | `[` or `]`                |
| Pause / Resume       | `Space`                   |
| Toggle follow camera | `F`                       |

---

## Command-Line Arguments

| Argument       | Description                                                                 |
| -------------- | --------------------------------------------------------------------------- |
| `--exptid`     | Experiment ID (default auto-generated)                                      |
| `--device`     | Device to use (e.g., `cuda:0`, `cpu`)                                       |
| `--delay`      | Whether to add perception delay                                             |
| `--checkpoint` | Specific checkpoint to load (default: latest)                               |
| `--resume`     | Resume from another checkpoint                                              |
| `--resumeid`   | Experiment ID to resume from                                                |
| `--seed`       | Random seed                                                                 |
| `--no_wandb`   | Disable Weights & Biases logging                                            |
| `--use_camera` | Enable camera or scan dots                                                  |
| `--web`        | Enable headless web visualization (requires VSCode Live Preview or browser) |

---

## 🙏 Acknowledgements

This repository builds upon and extends the following projects:

* [Extreme Parkour with Legged Robots (Cheng et al., 2023)](https://github.com/chengxuxin/extreme-parkour)
* [leggedrobotics/legged_gym](https://github.com/leggedrobotics/legged_gym)
* [Toni-SM/skrl](https://github.com/Toni-SM/skrl)

---

## Citation

If you find this project useful, please cite the original paper:

```bibtex
@article{cheng2023parkour,
  title={Extreme Parkour with Legged Robots},
  author={Cheng, Xuxin and Shi, Kexin and Agarwal, Ananye and Pathak, Deepak},
  journal={arXiv preprint arXiv:2309.14341},
  year={2023}
}
```

---

## License

This repository retains the same license as the original project (e.g., MIT License).
Modifications in this fork © 2025 JiaHe Yang
Original work © 2023 Xuxin Cheng, Kexin Shi, Ananye Agarwal, and Deepak Pathak.


# XTune - Model finetune tool for Intel GPU

**`Xtune`** is an model finetune tool for Intel GPU(Intel Arc 770)

> [!NOTE]
>
> - _`Xtune`_ incorporates with Llama-Factory to offer various methods for finetuning visual models (CLIP, AdaCLIP), LLM and Multi-modal models​. It makes easier to choose the method and to set fine-tuning parameters.

The core features include:

- Four finetune method for CLIP, details in [CLIP](./doc/key_features_for_clip_finetune_tool.md)
- Three finetune method for AdaCLIP, details in [AdaCLIP](./doc/adaclip_readme.md)
- Automatic hyperparameter searching enabled by Optuna [Optuna](https://github.com/optuna/optuna)
- Distillation from large models with Intel ARC GPU​
- Incorporate with Llama-Factory UI​
- Finetune methods for multi-modal models (to be supported)​

You can use this UI to easily access basic functions(merge two tool into one UI),

or use the command line to use tools separately which is easier to customize parameters and has more comprehensive functionality.

## Installation

Please install git first and make sure `git clone` can work.

Run install_xtune.sh to prepare component.

```bash
apt install -y rsync
bash prepare_xtune.sh
```

Then please fololow [install_dependency](./doc/install_dependency.md) to install Driver for Arc 770

> [!IMPORTANT]
> Installation is mandatory.

```bash
conda create -n xtune python=3.10 -y
conda activate xtune
pip install -r requirements.txt
# if you want to run on NVIDIA GPU
    conda install pytorch torchvision cudatoolkit=10.2 -c pytorch
# else run on A770
# You can refer to https://github.com/intel/intel-extension-for-pytorch for latest command
    python -m pip install torch==2.5.1+cxx11.abi torchvision==0.20.1+cxx11.abi torchaudio==2.5.1+cxx11.abi  --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/

cd src/llamafactory/clip_finetune/dassl
python setup.py develop
cd ../../../..
pip install matplotlib
pip install -e ".[metrics]"
pip install transformers==4.45.0 datasets==2.21.0
python -m pip install intel-extension-for-pytorch==2.5.10+xpu oneccl_bind_pt==2.5.0+xpu --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/
```

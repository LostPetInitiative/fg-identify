# Fine Grained Method for individual Identify

## Requirments

- PyTorch 1.0+
- timm
- scikit-learning 1.0+
- albumentations
- PyTorchLightning, should compatible with pytorch vesion

## Usage

1. Edit `config.yml` file, model_name is the backbone architecture name used in timm lib.
2. Run `train.py` with shell command, two specified arguments, config and device, must be passed. For example : `python ./fg_identify/train.py -c /home/kky/project/BoolArtProject/fg_identify/config.yml -d 0`
3. Run prediction & evaluation script, the detail are in `main.ipynb` file

## Experiments
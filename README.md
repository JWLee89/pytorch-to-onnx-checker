# pytorch-to-onnx-checker

A basic PyTorch to Onnx checker that is designed to

- Check the output difference between each nn.Module during forward pass.

For now, will start off small and maybe improve features based on needs

## Setup Virtual env

There is a simple script for setting up virtual env called `setup_venv.py`.
To run it, type in the following:

```Shell
# for dev
python setup_venv.py -r requirements/base.txt

# for dev + test. Name of venv folder is test_venv
# default is "venv"
python setup_venv.py -r requirements/base.txt -r requirements/check.txt
-v test_venv
```

Afterwards simple activate the venv by typing.

```Shell
source <venv-folder-name>/bin/activate
```

Enjoy developing / utilizing.

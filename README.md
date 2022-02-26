# Kernel Filtering Linear Overparameterization (KFLO)

The implementation in this repo was developed upon the [ACNet](https://arxiv.org/abs/1908.03930) code in the repo https://github.com/DingXiaoH/ACNet. Changes were made to implement and run KFLO.

## Instructions

Use CUDA==10.2 and install the following packages in your python environment using the following commands:
```
pip install torch==1.3.0
pip install torchvision==0.4.1
pip install h5py
pip install tensorflow-gpu==1.15.0
pip install coloredlogs
pip install tqdm
```

For CIFAR-10 experiments follow the steps below, **copied from https://github.com/DingXiaoH/ACNet** excluding the method related content:

1. Enter this directory.

2. Make a soft link to your CIFAR-10 directory. If the dataset is not found in the directory, it will be automatically downloaded.
```
ln -s YOUR_PATH_TO_CIFAR cifar10_data
```

3. Set the environment variables.
```
export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES=0
```

4. Train the Cifar-quick KFLO. Then train a regular model as baseline for the comparison.
```
python kflo/do_kflo.py -a cfqkbnc -b kflo
python kflo/do_kflo.py -a cfqkbnc -b base
```

5. Do the same on VGG.
```
python kflo/do_kflo.py -a vc -b kflo
python kflo/do_kflo.py -a vc -b base
```

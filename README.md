# This is the official repo of ["The Crack Diffusion Model: An Innovative Diffusion-Based Method for Pavement Crack Detection"](https://doi.org/10.3390/rs16060986).

Suggest train and test on Linux.

The command for training:

## On Single GPU

```
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.run --nproc_per_node=1 multi_task_train.py
```

## On Multiple GPU

```
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run --nproc_per_node=2 multi_task_train.py
```

Change from multi_task_train.py to multi_task_test.py for testing.

If train and test on Windows, 

remove notation
```# os.environ["CUDA_VISIBLE_DEVICES"]='0'  # if train on windows```
in _multi_task_train.py_ and
```# args.dist_backend = 'gloo'  # if train on windows```
in _distribute_utils.py_. 

Use

```
python -m torch.distributed.run --nproc_per_node=1 multi_task_train.py
```

# Target Propagation for Recurrent Neural Networks (TPRNN)

This is a self contained software accompanying the paper titled: Training
Language Models using Target-Propagation, available at https://arxiv.org/abs/1702.04770.
The code allows you to reproduce our results on two language modeling
datasets, Penntree Bank and Text8, using various training methods.


The code implements the following training algorithsm for RNNs:

- Standard BPTT training
- Penalty Method (PM)
- Alternating Direction Method of Multipliers (ADMM)
- Augmented Lagrangian Method (ALM)

It also allows you to play around with various hyper-parameters,
including the recurrent model architecture, learning rates and others.

## Examples
Here are some of the examples of how to use the code.

* To train a single layer LSTM with standard BPTT training for word-level language modeling on PenntreeBank dataset with following hyper-parameters:
  - hidden units: 100
  - minibatch size: 32
  - learning rate: 0.05

type:
```
th -i train_lm_bptt.lua -dset ptbw -model LSTM -nhid 100 -nlayer 1 batchsize 32 -lr 0.05
```

* To train a single layer GRU with ADMM for word-level language modeling on
Text8 with following hyper-parameters:

  - hidden units: 100
  - block size: 10
  - minibatch size: 32
  - parameter learning rate: 0.05
  - hidden learning rate: 1

type:
```
th -i train_lm_tprop.lua -dset text8w -model GRU -nhid 100 -block_size 10 -batchsz 32 -param_lr 0.05 -h_lr 1
```

* To train a single layer GRU with ALM for word-level language modeling on
Text8 with following hyper-parameters:
- hidden units: 100
- block size: 10
- minibatch size: 32
- parameter learning rate: 0.05
- hidden learning rate: 1

type:
```
th -i train_lm_tprop.lua -dset text8w -model GRU -nhid 100 -block_size 10 -batchsz 32 -param_lr 0.05 -h_lr 1 -alm
```

* To train a single layer GRU with PM for word-level language modeling on
Text8 with following hyper-parameters:
- hidden units: 100
- block size: 10
- minibatch size: 32
- parameter learning rate: 0.05
- hidden learning rate: 1

type:
```
th -i train_lm_tprop.lua -dset text8w -model GRU -nhid 100 -block_size 10 -batchsz 32 -param_lr 0.05 -h_lr 1 -u_startupdate 50000
```

To list all the options available, you need to type
```
th train_lm_bptt.lua --help
```
or

```
th train_lm_tprop.lua --help
```


## Requirements
The software requires you to have the following two packages already
installed on your systems:

- Torch 7
- cudnn
- torchnet
- Installation instructions for both on Ubuntu 14.04 are here: https://github.com/facebook/fbcunn/blob/master/INSTALL.md


## Installing
Download the files in an appropriate directory and run the code from there. See below.


## How Target Propagation for Recurrent Neural Networks Software works
The top level file for standard BPTT training is called train_lm_bptt.lua
The top level file for target-prop training is called train_lm_tprop.lua

In order to run the code you need to run the file using torch. For example:

```
th -i train_lm_bptt.lua -<option1_name> option1_val -<option2_name> option2_val ...
```
or

```
th -i train_lm_tprop.lua -<option1_name> option1_val -<option2_name> option2_val ...
```

In order to check what all options are available, type

```
th -i train_lm_bptt.lua --help
```
or

```
th -i train_lm_tprop.lua --help
```


## License
Target Propagation for Recurrent Neural Networks (TPRNN) is CC-NC licensed.


## Other Details
See the CONTRIBUTING file for how to help out.

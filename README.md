![PyTorch Logo](https://github.com/pytorch/pytorch/blob/master/docs/source/_static/img/pytorch-logo-dark.png)

--------------------------------------------------------------------------------

This is a modified pytorch to interpose memory allocation for features (weights and activations)
An additional flag indicates if the allocation function is for weights or activations.
You have to manually insert the flags on targeted functions

Hooked up memory allocations are handled in our custom memory allocation function.
The allocation maps are decided by an RL agent for DRAM and Persistent Memory.

Here are the functions that provide a flag for memory allcation interposition we developed so far.
- torch.empty(). (usage torch.empty(size, flag, ....,)
- torch.empty_like(). (usage torch.empty(size, flag, ....,)

We already embedded flags for Resnet
Here are the flagged network
- Linear()
- Sequential()
- 


## License
PyTorch has a BSD-style license, as found in the [LICENSE](LICENSE) file.

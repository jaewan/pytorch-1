![PyTorch Logo](https://github.com/pytorch/pytorch/blob/master/docs/source/_static/img/pytorch-logo-dark.png)

--------------------------------------------------------------------------------

This is a modified pytorch to interpose memory allocation for features (weights and activations)
An additional flag indicates if the allocation function is for weights or activations.
You have to manually insert the flags on targeted functions

Hooked up memory allocations are handled in our custom memory allocation function.
The allocation maps are decided by an RL agent for DRAM and Persistent Memory.
We used MARL to decide memory allocations. You can refer to the paper for the detail of MARL.

Usage:
To comply with multi-threads, two initializing function is used.
1. From the parent thread, call torch.hook_init(1, True);
2. From each thraeds, call torch.hook_init(2, True);
For a single thread, call both together before usage.
You can run non-interpositioning version without initializing functions.

Before running please make sure that you initialize the PM server as AppDirect Mode.
PMEM device should be named as /dev/dax0.0

We reommend mapping results directory as tmpfs for fast inter proceess communication
You can find scripts under scripts directory.

This version communicates with RL agent with socket. 
server.py will listen for any inference reqeust with memory mapping.
Once a mapping is arrived, the python script writes the mappings to a file in results/ directory which is
mounted as tmpfs.
At hook_init(1, false) call, pytorch reads the mappings and save it in an array.

Here are the functions that provide a flag for memory allcation interposition we developed so far.
- torch.empty(). (usage torch.empty(size, flag, ....,)
- torch.empty_like(). (usage torch.empty(size, flag, ....,)

We already embedded flags for Resnet. Any version including 3D can be deployed without any modification.
Here are the flagged network
- Linear()
- Sequential()


## Contact
If you have trouble-shooting or any questions regarding this project please contact
jaewan [at] berkeley.edu


## License
PyTorch has a BSD-style license, as found in the [LICENSE](LICENSE) file.

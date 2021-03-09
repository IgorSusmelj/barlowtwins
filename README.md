# barlowtwins
PyTorch Implementation of Barlow Twins paper: [Barlow Twins: Self-Supervised Learning via Redundancy Reduction](https://arxiv.org/pdf/2103.03230.pdf)


This is currently a work in progress. The code is a modified version of the SimSiam implementation [here](https://github.com/IgorSusmelj/simsiam-cifar10) 

- Time per epoch is around 40 seconds on a V100 GPU
- GPU usage is around 9 GBytes
- The current version reaches around 75% test accuracy

**Todo:**

- [X] warmup learning rate from 0
- [ ] report results on cifar-10
- [ ] create PR to add to lightly

### Installation

`pip install -r requirements.txt`

### Dependencies

- PyTorch
- PyTorch Lightning
- Torchvision
- lightly
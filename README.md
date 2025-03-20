Official implementation of Recurrent transformer variational autoencoders for multi-action motion synthesis [arxiv](https://arxiv.org/abs/2206.06741)

Built on top of [ACTOR](https://github.com/Mathux/ACTOR)

Other frameworks used: [fast-transformers](https://github.com/idiap/fast-transformers)

Dataset used: [PROX](https://prox.is.tue.mpg.de) 

To prepare the data, please use src/preprocess/prepare_proxmulti_dataset.py. It will process the sequences and their action labels, which reside in annotations.

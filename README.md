# Patch Based Semantic Segmention Project Template

> this segmention project template is rooted in my daily experimental code(you can find them in my repo)


# Project Struct

configs
-
**json file**: parameter table of this project

**utils**: some auxiliary functions

experiments
-
**data_loaders**: to generate hdf5 datasets 
> `experiments` folder is also the palce to store datasets/experiment log/network checkpoints etc.

perception
-
**base**: store template class of `data_loader` `trainer` `infer` `model`

**model**: store the definition of semantic segmention model

**infer**: how to predict

**trainer**: the way to train your model

main_trainer
-
starter of model training.First time you run the code,it will generate folder under `experiments folder`.This folder is where you store your `dataset` and find `log`,`model definition`

main_test
-
run this file to predict on experiments/(test_name)/test/origin/ data.

# Set up Environment
make sure your PC/Server has:

- python 3.5+
- tensorflow 1.4+
- keras 2.1.1+
- opencv_python 3.0+
- bund 1.2+

highly recommend installing anaconda to get python environment!

# Train a neural net

To start training with default options do `./run.lua` <br>

##Prepare data
You need to prepare data first. 
- To subsample imagenet data do: `th ../Train/Data/prepare_imagenet.lua --subsample_test true --subsample_train true --subsample_classes elab`
- To look at data, run `qlua ../Train/Data/prepare_imagenet.lua --show_test`
- See [`Data`](Data) for details.

##run options
You can specify options in `run.lua` file
- `opt.cuda=true` - if you want to use GPU
- `opt.mmload=true` - if you want to use memory mapping. Memory mapping allows to use datasets which are larger then RAM of your computer.

##Other scripts
- Models of networks are defined in the `models.lua` file.
- The training and testing routines are contained into the `train-and-test.lua` file.



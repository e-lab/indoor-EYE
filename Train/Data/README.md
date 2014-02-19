# Run from scratch
In order to *prepare* the dataset for being used by our training script we need the following files into the location `$EEX_DATASETS`

 - `train256m-data.t7`. Contains jpeg compressed train images. Data is stored as one dimensional ByteStorage.
 - `train256m-info.t7`. Contains train labels, offsets of compressed images in ByteStorage and their sizes.
 - `test256m-data.t7`. Contains jpeg compressed test images.
 - `test256m-info.t7`. Contains test labels, offsets of compressed images in ByteStorage and their sizes.
 - `classes.csv`. Contains names of classes.

## Prepare (subsample) data
```bash
th prepare-imagenet.lua --subsample_name elab --subsample_test --subsample_train --convert_class_names
```
Creates a subsample `elab` of imagenet. You can specify classes of your own subsample in file `indoor-classes.lua`. This will create the following files

 - `train-data-elab.t7`. Jpeg compressed subsample train images. Data is stored as one dimensional ByteStorage.
 - `train-info-elab.t7`. Labels, sizes and offsets of train subsample. 
 - `test-data-elab.t7`. Test images
 - `test-info-elab.t7`. Test labels, sizes, offsets.
 - `classes.th`. A map from class id to class name.

## Look at data
- `qlua prepare-imagenet.lua --show_test`. Displays test images of each class on a separate figure. 
- `qlua prepare-imagenet.lua --show_train`. Displays train images of each class on a separate figure. 
- `qlua prepare-imagenet.lua --save_test`. Saves test images of each class in a separate file. 
- `qlua prepare-imagenet.lua --save_train`. Saves train images of each class in a separate file. 

## Functions for loading data
- `load_data()`. Loads and decompress jpegs images, resizes them and do global normalization (subtract mean and div by std). Loads labels. Defines `prepareBatch()` and `copyBatch()` functions. All data is loaded to RAM. 
- `load_data_mm()`. Loads labels, jpeg sizes and jpeg offsets. Opens jpeg data file for memory-mapped reading. Defines `prepareBatch()` and `copyBatch()` functions.

## Load batch of images
- `prepareBatch(idx, istest)`. Loads batch `idx` in RAM and normilize globally (subtract global mean and divide by global std). In case of memory-mapping it is done asynchronously. 
- `copyBatch()`. Returns last prepared batch.  

## Train model
Eventually we can set up our network's architecture (see [`models.lua`](../models.lua)) and
```bash
./run.lua
```

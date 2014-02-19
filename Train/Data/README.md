# Run from scratch
In order to *prepare* the dataset for being used by our training script we need the following files into the location `$EEX_DATASETS`

 - `train256m-data.t7`
 - `train256m-info.t7`
 - `test256m-data.t7`
 - `test256m-info.t7`
 - `classes.csv`

## Prepare (subsample) data
```bash
torch prepare-imagenet.lua --subsample_test --subsample_train --convert_class_names
```
This will create the following files

 - `train-data-elab.t7`
 - `train-info-elab.t7`
 - `test-data-elab.t7`
 - `test-info-elab.t7`
 - `classes.th`

## Look at data
```bash
torch prepare-imagenet.lua --show_test
```

## Train model
Eventually we can set up our network's architecture (see [`models.lua`](../models.lua)) and
```bash
./run.lua
```

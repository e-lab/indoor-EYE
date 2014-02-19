# Run from scratch
In order to *prepare* the dataset for being used by our training script we need the several files into the location `$EEX_DATASETS`.

 - `train256m-data.t7`
 - `train256m-info.t7`
 - `test256m-data.t7`
 - `test256m-info.t7`
 - `classes.csv`
 
Subsample data. <br> `cd Train/Data` <br> `th prepare-imagenet.lua --subsample_test true --subsample_train true --convert_class_names true` <br> This will create files `train-data-elab.t7`, `train-info-elab.t7`, `test-data-elab.t7`, `test-info-elab.t7`, `classes.th`.
Look at data. <br> `qlua prepare-imagenet.lua --show_test true`
Train model. <br> `cd ../` (Train folder)  
`th run.lua`
Run demo. <br> `qlua run.lua -v ../videos/eLab.mp4 -downsampling 1 -x 0`

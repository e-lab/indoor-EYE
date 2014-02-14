# indoor-EYE

Object recognition application for indoor environment

## Repository organisation
The repository contains main folders having a name starting with a capital. These are folders that contain the main code and are sync with the main repository.  
Running the code will generate other temporary folders, having lowercase names, and those are only local to your machine.  
The repository tree, at the moment, is the following
```
.
├── Demo
├── Train
├───── Data
├───── results
├───── temp-data
└── README.md
```

### Main folders and files
 - [`Demo`](Demo): hosts the demonstration routine, runnable on images or videos 
 - [`Train`](Train): encloses the main program, `run`, which trains the network with several specificable options
 - [`Train/Data`](Train/Data): scripts for preparing and processing imagenet data
 - [`README.md`](README.md): this file :)

### Local temporary folders and files
 - `Train/results`: contains the trained network and data for preprocessing
 - `Train/temp-data`: stores all temporary data
 
### Run from scratch
- Prepare data: `cd Train/Data`, `th prepare_imagenet.lua --subsample_test true --subsample_train true --subsample_classes elab`
- Look at data: `qlua prepare_imagenet.lua --show_test`
- Train model: `cd Train`,  `th run.lua --train_data_file train-data-elab.t7 --train_info_file train-info-elab.t7 --test_data_file test-data-elab.t7 --test_info_file test-info-elab.t7 --subsample_classes elab`
- Run demo: `qlua run.lua -v ../videos/eLab.mp4 -downsampling 1 -x 0`
 

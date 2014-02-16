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
- Data. Required files in `(eex.datasetsPath() .. 'imagenet2012/')` folder: `train256m-data.t7`, `train256m-info.t7`, `test256m-data.t7`, `test256m-info.t7`, `classes.csv`.  
- Subsample data. <br> `cd Train/Data` <br> `th prepare-imagenet.lua --subsample_test true --subsample_train true --convert_class_names true` <br> This will create files `train-data-elab.t7`, `train-info-elab.t7`, `test-data-elab.t7`, `test-info-elab.t7`, `classes.th`.
- Look at data. <br> `qlua prepare-imagenet.lua --show_test true`
- Train model. <br> `cd ../` (Train folder)  
`th run.lua`
- Run demo. <br> `qlua run.lua -v ../videos/eLab.mp4 -downsampling 1 -x 0`
 

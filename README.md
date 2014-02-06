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
 - [`README.md`](README.md): this file :)

### Local temporary folders and files
 - `Train/results`: contains the trained network and data for preprocessing
 - `Train/temp-data`: stores all temporary data

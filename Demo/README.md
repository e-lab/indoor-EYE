# Demo
## You need to build the model first
You need to train the model before you can run the demo. For details see [`Train`](../Train).

## Run demo
```bash
qlua run.lua -v ../videos/eLab.mp4 -downsampling 1 -x 0
```

## Run `Top5-Demo`
`Top5-Demo` can show top 5 model's prediction on the testing dataset (default option) and on a live camera video. To print a list of options, run

```bash
torch Top5-Demo.lua --help
```

To run the demo on the testing set, you can simply run `torch Top5-Demo.lua`. To get input from camera, we can run it as following

```bash
torch Top5-Demo.lua --camera --fps 3
```

In order to print the *histogram* of the prediction probabilities, you need just to include the `--histogram` flag.

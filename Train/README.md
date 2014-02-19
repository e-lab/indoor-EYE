# Train a neural net

To start training with default options do `./run.lua`

## Prepare data
You need to prepare data first. See [`Data`](Data) for details.

## Available options
You can see the list of available options by running

```bash
./run --help
```
Among the others, we have that

 - `--cuda`: trains using your GPU;
 - `--mmload`: uses memory mapping. Memory mapping allows to use datasets which are larger then RAM of your computer.

## Network model
Models of networks are defined in the `models.lua` file.

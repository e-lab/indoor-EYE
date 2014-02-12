-------------------------------------------------------------------------------
-- Define neuralnet models
-- Artem Kuharenko
-------------------------------------------------------------------------------
function get_model1()

   --options for (conv+pool+threshold) layers
   local nlayers = 2 --number of (conv+pool+threshold) layers
   local nfeatures= {[0]=3, 32, 32} --number of feature maps in conv layers
   local filtsizes = {7, 5} --filter sizes in conv layers
   local paddings = {0, 0} 
   local strides = {1, 1}
   local poolsizes = {2, 2}

   --options for linear layers
   local linears = {32, #classes} --number of neurons in linear layer

   printable_layers = {2} --layers for printing weights stat
   
   --define neural net
   local model = nn.Sequential()

   --transpose batch if cuda
   if opt.cuda then
      model:add(nn.Transpose({1,4},{1,3},{1,2}))
   end
   
   local mapsizes = {[0]=opt.width} --sizes of output of layers
   local nconnections = {[0]=0} --number of connections between i-th and (i-1) layer
   local nhiddens = {[0] = 0} --number of hidden units in layer i
   local nouts = {[0] = opt.width^2 * opt.ncolors} --number of output units in layer i

   --add first 1..nlayers layers (conv+pool+threshold)
   for i = 1, nlayers do

      local conv_layer = {}
      local pool_layer = {}
      local test_batch = torch.Tensor(64, nfeatures[i - 1], mapsizes[i - 1], mapsizes[i - 1])

      if opt.cuda then

         conv_layer = nn.SpatialConvolutionCUDA(nfeatures[i - 1], nfeatures[i], filtsizes[i], filtsizes[i], strides[i], strides[i], paddings[i])
         pool_layer = nn.SpatialMaxPoolingCUDA(poolsizes[i], poolsizes[i], poolsizes[i], poolsizes[i])
         conv_layer:cuda()
         pool_layer:cuda()
         test_batch = nn.Transpose({1,4},{1,3},{1,2}):forward(test_batch):cuda()

      else

         conv_layer = nn.SpatialConvolution(nfeatures[i - 1], nfeatures[i], filtsizes[i], filtsizes[i])
         pool_layer = nn.SpatialMaxPooling(poolsizes[i], poolsizes[i], poolsizes[i], poolsizes[i])

      end

      conv_layer.printable = true
      conv_layer.text = 'Conv layer ' .. i
      model:add(conv_layer)
      model:add(pool_layer)
      model:add(nn.Threshold(0,0))

      --get layer sizes
      local r1 = conv_layer:forward(test_batch)
      local r2 = pool_layer:forward(r1)
--      print(#conv_layer.output)
--      print(#pool_layer.output)
      mapsizes[i] = pool_layer.output:size(2)
      nhiddens[i] = conv_layer.weight:size(2) * conv_layer.weight:size(3) * conv_layer.weight:size(4)
      nconnections[i] = conv_layer.weight:size(1) * nhiddens[i] * (mapsizes[i - 1] - filtsizes[i] + 1) ^ 2
      nouts[i] = pool_layer.output:size(2) * pool_layer.output:size(3) * nfeatures[i]

   end

   --transpose batch if cuda
   if opt.cuda then
      model:add(nn.Transpose({4,1},{4,2},{4,3}))
   end

   --reshape
   model:add(nn.Reshape(nouts[nlayers]))

   --add linear layers
   for i = 1, #linears do

      nouts[nlayers + i] = linears[i]
      local linear_layer = nn.Linear(nouts[nlayers + i - 1], nouts[nlayers + i])
      linear_layer.printable = true
      linear_layer.text = 'Linear layer' .. i
      model:add(linear_layer)
      model:add(nn.Threshold(0, 0))

      --get layer sizes
      mapsizes[nlayers + i] = 1
      nhiddens[nlayers + i] = linears[i]
      nconnections[nlayers + i] = nouts[nlayers + i - 1] * nouts[nlayers + i]
      nfeatures[nlayers + i] = 1

   end

   --log probabilities
   model:add(nn.LogSoftMax())

   -- Loss: NLL
   local loss = nn.ClassNLLCriterion()

   if opt.cuda then
      model:cuda()
      loss:cuda()
   end

   --print sizes of nlayers
   for i = 0, nlayers + #linears do

      print(string.format(
      '==> model layer %02d  -  spatial extent: %03dx%03d  |  feature maps: %04d  |  hidden units: %05d  |  output size: %05d  |  connections: %05d',
      i, mapsizes[i], mapsizes[i], nfeatures[i], nhiddens[i], nouts[i], nconnections[i]
      ))

   end

  -- print(model.modules)
   return model, loss

end

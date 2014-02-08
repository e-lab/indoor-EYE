-------------------------------------------------------------------------------
-- Define neuralnet models
-- Artem Kuharenko
-------------------------------------------------------------------------------

function get_output_size(model)

   local tb = torch.Tensor(opt.batchSize, opt.ncolors, opt.height, opt.width)
   local res = {}

   if opt.cuda then
      local tb_cuda = tb:cuda()
      model:cuda()
      res = model:forward(tb_cuda)
   else
      res = model:forward(tb)
   end
   return res:size(3)

end

function get_model1()

   local mapsizes = {[0] = opt.width}
   local nfeatures= {[0]=3, 32, 32}
   local filtsizes = {7, 5}
   local paddings = {0, 0}
   local strides = {1, 1}
   local poolsizes = {2, 2}
   local nlayers = 2

   -- map sizes:
   for i = 1,#nfeatures do

      if filtsizes[i] == 1 then

         mapsizes[i] = 1

      else

         mapsizes[i] = mapsizes[i-1] + paddings[i] - filtsizes[i] + 1

         if strides[i] then
            mapsizes[i] = mapsizes[i] / strides[i]
         end

         if poolsizes[i] then
            mapsizes[i] = mapsizes[i] / poolsizes[i]
         end

         mapsizes[i] = math.floor(mapsizes[i])

      end

   end

   -- nb of hidden units per layer:
   local nunits = {}
   for k,mapsize in pairs(mapsizes) do
      nunits[k] = mapsizes[k]^2 * nfeatures[k]
   end

   local nconnections = {[0]=0}
   for k = 1, nlayers do
      nconnections[k] = filtsizes[k]^2 * nfeatures[k - 1] * nfeatures[k] * (mapsizes[k] - filtsizes[k] + 1) ^ 2
   end

   --[[   for i = 0,#mapsizes do

   print(string.format(
   '==> model layer %02d  -  spatial extent: %03dx%03d  |  unique features: %04d  |  hidden units: %05d  |  connections: %05d',
   i, mapsizes[i], mapsizes[i], nfeatures[i], nunits[i], nconnections[i]
   ))

   end
   --]]

   local model = nn.Sequential()

   if opt.cuda then
      model:add(nn.Transpose({1,4},{1,3},{1,2}))
   end

   local realsizes = {[0]=opt.width}
   --stages 1..nlayers
   for i = 1, nlayers do
      --add layer i: convolution, maxpooling, threshold
      if opt.cuda then

         model:add(nn.SpatialConvolutionCUDA(nfeatures[i - 1], nfeatures[i], filtsizes[i], filtsizes[i], strides[i], strides[i], paddings[i]))
         model:add(nn.SpatialMaxPoolingCUDA(poolsizes[i], poolsizes[i], poolsizes[i], poolsizes[i]))

      else

         model:add(nn.SpatialConvolution(nfeatures[i - 1], nfeatures[i], filtsizes[i], filtsizes[i]))
         model:add(nn.SpatialMaxPooling(poolsizes[i], poolsizes[i], poolsizes[i], poolsizes[i]))

      end
      model:add(nn.Threshold(0,0))

      realsizes[i] = get_output_size(model)

   end

   if opt.cuda then
      model:add(nn.Transpose({4,1},{4,2},{4,3}))
   end

   for i = 0, nlayers do

      print(string.format(
      '==> model layer %02d  -  spatial extent: %03dx%03d  |  unique features: %04d  |  hidden units: %05d  |  connections: %05d',
      i, realsizes[i], realsizes[i], nfeatures[i], nunits[i], nconnections[i]
      ))

   end

   --linear
   model:add(nn.Reshape(nunits[nlayers]))
   model:add(nn.Linear(nunits[nlayers], opt.classifier_size))
   model:add(nn.Threshold())

   --linear (classifier)
   model:add(nn.Linear(opt.classifier_size, #classes))

   --log probabilities
   model:add(nn.LogSoftMax())

   -- Loss: NLL
   local loss = nn.ClassNLLCriterion()

   model:cuda()
   loss:cuda()

   return model, loss

end

function get_model1()

   local mapsizes = {[0] = opt.width}
   local nfeatures= {[0]=3, 32, opt.classifier_size, #classes}
   local filtsizes = {7, 1, 1}
   local paddings = {0, 0, 0}
   local strides = {1, 1, 1}
   local poolsizes = {2}

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

   for i = 0,#mapsizes do
    
     print(string.format(
         '==> model layer %02d  -  spatial extent: %03dx%03d  |  unique features: %04d  |  hidden units: %05d',
         i, mapsizes[i], mapsizes[i], nfeatures[i], nunits[i]
      ))
  
   end

   local model = nn.Sequential()
   
   if opt.cuda then 
      model:add(nn.Transpose({1,4},{1,3},{1,2}))
   end

   --stage 1
   if opt.cuda then

      model:add(nn.SpatialConvolutionCUDA(opt.ncolors, nfeatures[1], filtsizes[1], filtsizes[1]))
      model:add(nn.SpatialMaxPoolingCUDA(poolsizes[1], poolsizes[1], poolsizes[1], poolsizes[1]))

   else
   
      model:add(nn.SpatialConvolution(opt.ncolors, nfeatures[1], filtsizes[1], filtsizes[1]))
      model:add(nn.SpatialMaxPooling(poolsizes[1], poolsizes[1], poolsizes[1], poolsizes[1]))

   end
   model:add(nn.Threshold(0,0))
   
   if opt.cuda then
      model:add(nn.Transpose({4,1},{4,2},{4,3}))
   end

   --stage 2: linear 
   model:add(nn.Reshape(nunits[1]))
   model:add(nn.Linear(nunits[1], opt.classifier_size))
   model:add(nn.Threshold())
   
   --stage 3: linear (classifier)
   model:add(nn.Linear(opt.classifier_size, #classes))

	--log probabilities
   model:add(nn.LogSoftMax())

   -- Loss: NLL
   local loss = nn.ClassNLLCriterion()

   model:cuda()
   loss:cuda()

   return model, loss

end

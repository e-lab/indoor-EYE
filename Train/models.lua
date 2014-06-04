-------------------------------------------------------------------------------
-- Define neuralnet models
-- Artem Kuharenko
-- Alfredo Canziani, Mar 2014
-------------------------------------------------------------------------------

-- Craft model ----------------------------------------------------------------
function get_model1()

   --options for (conv+pool+threshold) layers
   local nConvLayers = 5 --number of (conv+pool+threshold) layers
   local nFeatureMaps= {[0]=3, 96, 256, 384, 384, 256} --number of feature maps in conv layers
   local filterSize  = {       11,   5,   3,   3,   3} --filter sizes in conv layers
   local convPadding = {        0,   0,   0,   0,   0}
   local convStride  = {        4,   1,   1,   1,   1}
   local poolSize    = {        2,   2,   1,   1,   2}
   local poolStride  = {        2,   2,   1,   1,   2}

   --options for linear layers
   local neuronsPerLinearLayer = {4096, 4096} --number of neurons in linear layer

   --neuralnet model consists of submodel1 and submodel2
   local model = nn.Sequential()
   local submodel1 = nn.Sequential() --conv+pool+threshold layers
   local submodel2 = nn.Sequential() --linear layers

   -- Keeping track of memory usage
   local memory = {}
   memory[0] = opt.batchSize * (#classes + opt.ncolors*opt.width^2) -- gradInput + output (overhead)
   memory.submodel1 = {}
   memory.submodel1.val = {}
   memory.submodel1.str = {}
   memory.submodel1.val[0] = opt.batchSize * opt.ncolors * opt.width^2 -- + output
   memory.submodel2 = {}
   memory.submodel2.val = {}
   memory.submodel2.str = {}

   -- Dropout in the input space
   local dropout = {}
   local DOidx = 1
   if opt.inputDO > 0 then
      dropout[DOidx] = nn.Dropout(opt.inputDO)
      submodel1:add(dropout[DOidx])
      DOidx = DOidx + 1
      table.insert(memory.submodel1.val,4 * opt.batchSize * opt.ncolors * opt.width^2)
      table.insert(memory.submodel1.str,'Drp')
   end

   --transpose batch if cuda
   if opt.cuda then
      submodel1:add(nn.Transpose({1,4},{1,3},{1,2}))
      table.insert(memory.submodel1.val,2 * opt.batchSize * opt.ncolors * opt.width^2)
      table.insert(memory.submodel1.str,'Trn')
   end

   local mapsizes = {[0]=opt.width} --sizes of output of layers
   local nConnections = {[0]=0} --number of connections between i-th and (i-1) layer
   local nUniqueWeights = {[0] = 0} --number of hidden units in layer i
   local nHiddenNeurons = {[0] = opt.width^2 * opt.ncolors} --number of output units in layer i
   local r1, r2

   --add first 1..nConvLayers layers (conv+pool+threshold)
   for i = 1, nConvLayers do

      local test_batch = torch.Tensor(opt.batchSize, nFeatureMaps[i - 1], mapsizes[i - 1], mapsizes[i - 1])

      local convLayer, poolLayer

      if opt.cuda then

         convLayer = nn.SpatialConvolutionCUDA(nFeatureMaps[i - 1], nFeatureMaps[i], filterSize[i], filterSize[i], convStride[i], convStride[i], convPadding[i])
         poolLayer = nn.SpatialMaxPoolingCUDA(poolSize[i], poolSize[i], poolStride[i], poolStride[i])
         convLayer:cuda()
         poolLayer:cuda()
         test_batch = nn.Transpose({1,4},{1,3},{1,2}):forward(test_batch):cuda()

      else

         convLayer = nn.SpatialConvolutionMM(nFeatureMaps[i - 1], nFeatureMaps[i], filterSize[i], filterSize[i])
         -- convLayer = nn.SpatialConvolution(nFeatureMaps[i - 1], nFeatureMaps[i], filterSize[i], filterSize[i], convStride[i], convStride[i])
         poolLayer = nn.SpatialMaxPooling(poolSize[i], poolSize[i], poolStride[i], poolStride[i])

      end

      --get layer sizes
      r1 = convLayer:forward(test_batch)
      r2 = poolLayer:forward(r1)

      convLayer.printable = true
      convLayer.text = 'Conv layer ' .. i
      submodel1:add(convLayer)

      -- Computing memory usage
      local biasMem = 2*nFeatureMaps[i]
      local weightMem = nFeatureMaps[i]*nFeatureMaps[i - 1]*filterSize[i]^2
      weightMem = opt.cuda and 3*weightMem or 2*weightMem
      local gradInputMem = test_batch:size(1)*test_batch:size(2)*test_batch:size(3)*test_batch:size(4)
      local outputMem = r1:size(1)*r1:size(2)*r1:size(3)*r1:size(4)
      table.insert(memory.submodel1.val, biasMem + weightMem + gradInputMem + outputMem)
      table.insert(memory.submodel1.str,'Cnv')

      if opt.probe then
         submodel1:add(nn.Probe('Probing ' .. convLayer.text))
      end

      if poolSize[i] > 1 then
         submodel1:add(poolLayer)
         table.insert(memory.submodel1.val, outputMem + r2:size(1)*r2:size(2)*r2:size(3)*r2:size(4))
         table.insert(memory.submodel1.str,'Pol')
         outputMem = r2:size(1)*r2:size(2)*r2:size(3)*r2:size(4)
      end

      if opt.cuda then
         mapsizes[i] = poolLayer.output:size(2)
      else
         mapsizes[i] = poolLayer.output:size(3)
      end
      nUniqueWeights[i] = convLayer.weight:size(1) * convLayer.weight:size(2)
      if opt.cuda then
         nUniqueWeights[i] = nUniqueWeights[i] * convLayer.weight:size(3) * convLayer.weight:size(4)
      end
      nConnections[i] = nUniqueWeights[i] * ((mapsizes[i - 1] - filterSize[i] + 1) / convStride[i]) ^ 2

      if opt.cuda then
         nHiddenNeurons[i] = poolLayer.output:size(2) * poolLayer.output:size(3) * nFeatureMaps[i]
      else
         nHiddenNeurons[i] = poolLayer.output:size(3) * poolLayer.output:size(4) * nFeatureMaps[i]
      end

      submodel1:add(nn.Threshold(0,0))
      table.insert(memory.submodel1.val, 2 * outputMem)
      table.insert(memory.submodel1.str,'NL')
   end

   --transpose batch if cuda
   if opt.cuda then
      submodel1:add(nn.Transpose({4,1},{4,2},{4,3}))
      table.insert(memory.submodel1.val, 2 * r2:size(1)*r2:size(2)*r2:size(3)*r2:size(4))
      table.insert(memory.submodel1.str,'Trn')
   end

   memory.submodel1.val[0] = memory.submodel1.val[0] + r2:size(1)*r2:size(2)*r2:size(3)*r2:size(4)

   memory.submodel2.val[0] = r2:size(1)*r2:size(2)*r2:size(3)*r2:size(4) + opt.batchSize*#classes
   --reshape
   submodel2:add(nn.Reshape(nHiddenNeurons[nConvLayers]))
   table.insert(memory.submodel2.val, 2 * nHiddenNeurons[nConvLayers] * opt.batchSize)
   table.insert(memory.submodel2.str,'Rsh')

   -- If dropout is not 0
   if opt.dropout > 0 then
      dropout[DOidx] = nn.Dropout(opt.dropout)
      submodel2:add(dropout[DOidx])
      DOidx = DOidx + 1
      table.insert(memory.submodel2.val, 4 * nHiddenNeurons[nConvLayers] * opt.batchSize)
      table.insert(memory.submodel2.str,'Drp')
   end

   --add linear layers
   for i = 1, #neuronsPerLinearLayer do

      nHiddenNeurons[nConvLayers + i] = neuronsPerLinearLayer[i]
      local linear_layer = nn.Linear(nHiddenNeurons[nConvLayers + i - 1], nHiddenNeurons[nConvLayers + i])
      linear_layer.printable = true
      linear_layer.text = 'Linear layer ' .. i
      submodel2:add(linear_layer)

      -- Computing memory usage
      local biasMem = 2 * nHiddenNeurons[nConvLayers + i]
      local weightMem = 2 * nHiddenNeurons[nConvLayers + i - 1] * nHiddenNeurons[nConvLayers + i]
      local gradInputMem = opt.batchSize * nHiddenNeurons[nConvLayers + i - 1]
      local outputMem = opt.batchSize * nHiddenNeurons[nConvLayers + i]
      table.insert(memory.submodel2.val, biasMem + weightMem + gradInputMem + outputMem)
      table.insert(memory.submodel2.str,'Lnr')

      if opt.probe then
         submodel1.val:add(nn.Probe('Probing ' .. linear_layer.text))
      end

      submodel2:add(nn.Threshold(0, 0))
      table.insert(memory.submodel2.val, 2 * outputMem)
      table.insert(memory.submodel2.str,'NL')

      -- If dropout is not 0
      if opt.dropout > 0 then
         dropout[DOidx] = nn.Dropout(opt.dropout)
         submodel2:add(dropout[DOidx])
         DOidx = DOidx + 1
         table.insert(memory.submodel2.val, 4 * outputMem)
         table.insert(memory.submodel2.str,'Drp')
      end

      --get layer sizes
      mapsizes[nConvLayers + i] = 1
      nUniqueWeights[nConvLayers + i] = neuronsPerLinearLayer[i]
      nConnections[nConvLayers + i] = nHiddenNeurons[nConvLayers + i - 1] * nHiddenNeurons[nConvLayers + i]
      nFeatureMaps[nConvLayers + i] = 1

   end

   --add classifier
   local outputLayer = nn.Linear(nHiddenNeurons[nConvLayers + #neuronsPerLinearLayer], #classes)
   outputLayer.printable = true
   outputLayer.text = 'Output layer'
   submodel2:add(outputLayer)

   -- Computing memory usage
   local biasMem = 2 * #classes
   local weightMem = 2 * nHiddenNeurons[#nHiddenNeurons] * #classes
   local gradInputMem = opt.batchSize * nHiddenNeurons[#nHiddenNeurons]
   local outputMem = opt.batchSize * #classes
   table.insert(memory.submodel2.val, biasMem + weightMem + gradInputMem + outputMem)
   table.insert(memory.submodel2.str,'Lnr')

   if opt.probe then
      submodel1.val:add(nn.Probe('Probing output layer'))
   end
   --log probabilities
   submodel2:add(nn.LogSoftMax())
   table.insert(memory.submodel2.val, 2 * outputMem)
   table.insert(memory.submodel2.str,'SM')

   --add submodels to model
   model:add(submodel1)
   model:add(submodel2)

   -- Loss: NLL
   local loss = nn.ClassNLLCriterion()

   if opt.cuda then
      model:cuda()
      loss:cuda()
   end

   --print sizes of nConvLayers
   statFile:write('\n==> Network model\n')
   for i = 0, nConvLayers + #neuronsPerLinearLayer do

      if i == 0 then filterSize[0] = 0
      elseif filterSize[i] == nil then filterSize[i] = 1 end

      local s = string.format(
      '==> model layer %2d  -  filter size: %2d  |  spatial extent: %3dx%3d  |  feature maps: %3d  |  hidden neurons: %6d  |  unique weights: %5d  |  connections: %9d\n',
      i, filterSize[i], mapsizes[i], mapsizes[i], nFeatureMaps[i], nHiddenNeurons[i], nUniqueWeights[i], nConnections[i]
      )
      if opt.verbose then io.write(s) end
      statFile:write(s)
      statFile:flush()

   end

   -- Evaluate network's weight
   local w = model:getParameters()
   print(string.format("The network's weights weight %0.2f MB", w:size(1)*4/1024^2))


   --print(model.modules)
   return model, loss, dropout, memory

end

-- Useful functions for <get_model2> -------------------------------------------
-- Iterative function definition for recovering the dropout table
local function assignDropout(dropoutTable,module)
   if module.__typename == 'nn.Dropout' then
      table.insert(dropoutTable, module)
   end
end

function recoverDropoutTable(dropoutTable,network)
   assignDropout(dropoutTable,network)
   if network.modules then
      for _,a in ipairs(network.modules) do
         recoverDropoutTable(dropoutTable,a)
      end
   end
end

-- Load model from file --------------------------------------------------------
function get_model2(networkFile)

   -- Load model from file
   local model = netToolkit.loadNet(networkFile)

   -- Recover the dropout pointers
   local dropout = {}
   recoverDropoutTable(dropout,model)

   -- Loss: NLL
   local loss = nn.ClassNLLCriterion()

   if opt.cuda then
      model:cuda()
      loss:cuda()
   end

   return model, loss, dropout

end

-------------------------------------------------------------------------------
-- Define neuralnet models
-- Artem Kuharenko
-- Alfredo Canziani, Mar 2014
-------------------------------------------------------------------------------

-- Craft model ----------------------------------------------------------------
function get_model1(nbClasses, statFile, cuda)

   --options for (conv+pool+threshold) layers
   local nConvLayers = 12 --number of (conv+pool+threshold) layers
   local nFeatureMaps= {[0]=3, 16, 16, 16, 32, 32, 32, 32, 32, 32, 32, 32, 32} --number of feature maps in conv layers
   local filterSize  = {        3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3} --filter sizes in conv layers
   local convPadding = {        0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0}
   local convStride  = {        1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1}
   local poolSize    = {        1,  1,  2,  1,  1,  2,  1,  1,  2,  1,  1,  2}
   local poolStride  = {        1,  1,  2,  1,  1,  2,  1,  1,  2,  1,  1,  2}

   --options for linear layers
   local neuronsPerLinearLayer = {128, 128} --number of neurons in linear layer


   --neuralnet model consists of submodel1 and submodel2
   local model = nn.Sequential()
   local submodel1 = nn.Sequential() --conv+pool+threshold layers
   local submodel2 = nn.Sequential() --linear layers

   -- Keeping track of memory usage
   local memory = {}
   memory[0] = opt.batchSize * (nbClasses + 3*opt.side^2) -- gradInput + output (overhead)
   memory.submodel1 = {}
   memory.submodel1.val = {}
   memory.submodel1.str = {}
   memory.submodel1.val[0] = opt.batchSize * 3 * opt.side^2 -- + output
   memory.submodel2 = {}
   memory.submodel2.val = {}
   memory.submodel2.str = {}

   -- Dropout in the input space
   local dropout = {}
   local DOidx = 1
   if opt.inputDO > 0 then
      dropout[DOidx] = nn.Dropout(opt.inputDO)
      dropout[DOidx].updateGradInput = function () return nil end
      submodel1:add(dropout[DOidx])
      DOidx = DOidx + 1
      table.insert(memory.submodel1.val,4 * opt.batchSize * 3 * opt.side^2)
      table.insert(memory.submodel1.str,'Drp')
   end

   local mapsizes = {[0]=opt.side} --sizes of output of layers
   local nConnections = {[0]=0} --number of connections between i-th and (i-1) layer
   local trainParam = {[0] = 0} --number of hidden units in layer i
   local nHiddenNeurons = {[0] = opt.side^2 * 3} --number of output units in layer i
   local r1, r2

   --add first 1..nConvLayers layers (conv+pool+threshold)
   for i = 1, nConvLayers do

      local test_batch = torch.Tensor(opt.batchSize, nFeatureMaps[i - 1], mapsizes[i - 1], mapsizes[i - 1])
      local convLayer, poolLayer

      if cuda then
         convLayer = nn.SpatialConvolutionMM(nFeatureMaps[i - 1], nFeatureMaps[i], filterSize[i], filterSize[i], convStride[i], convStride[i], convPadding[i])
         poolLayer = nn.SpatialMaxPooling(poolSize[i], poolSize[i], poolStride[i], poolStride[i])
         convLayer:cuda()
         poolLayer:cuda()
         test_batch= test_batch:cuda()
      else
         convLayer = nn.SpatialConvolutionMM(nFeatureMaps[i - 1], nFeatureMaps[i], filterSize[i], filterSize[i])
         poolLayer = nn.SpatialMaxPooling(poolSize[i], poolSize[i], poolStride[i], poolStride[i])
      end
      --get layer sizes

      r1 = convLayer:forward(test_batch)
      r2 = poolLayer:forward(r1)

      convLayer.printable = true
      convLayer.text = 'Conv layer ' .. i
      if (i == 1) then
         convLayer.updateGradInput = function () return nil end
      end
      submodel1:add(convLayer)

      -- Computing memory usage
      local biasMem = 2*nFeatureMaps[i]
      local weightMem = nFeatureMaps[i]*nFeatureMaps[i - 1]*filterSize[i]^2
      weightMem = cuda and 3*weightMem or 2*weightMem
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

      mapsizes[i] = poolLayer.output:size(3)

      trainParam[i] = convLayer.weight:size(1) * convLayer.weight:size(2)

      trainParam[i] = trainParam[i] + nFeatureMaps[i]
      nConnections[i] = trainParam[i] * ((mapsizes[i - 1] - filterSize[i] + 1) / convStride[i]) ^ 2

      nHiddenNeurons[i] = poolLayer.output:size(3) * poolLayer.output:size(4) * nFeatureMaps[i]

      submodel1:add(nn.ReLU())
      table.insert(memory.submodel1.val, 2 * outputMem)
      table.insert(memory.submodel1.str,'NL')
   end

   memory.submodel1.val[0] = memory.submodel1.val[0] + r2:size(1)*r2:size(2)*r2:size(3)*r2:size(4)

   memory.submodel2.val[0] = r2:size(1)*r2:size(2)*r2:size(3)*r2:size(4) + opt.batchSize*nbClasses
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

      submodel2:add(nn.ReLU())
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
      trainParam[nConvLayers + i] = nHiddenNeurons[nConvLayers + i - 1] * nHiddenNeurons[nConvLayers + i]
      nConnections[nConvLayers + i] = nHiddenNeurons[nConvLayers + i - 1] * nHiddenNeurons[nConvLayers + i]
      nFeatureMaps[nConvLayers + i] = 1

   end

   --add classifier
   local outputLayer = nn.Linear(nHiddenNeurons[nConvLayers + #neuronsPerLinearLayer], nbClasses)
   outputLayer.printable = true
   outputLayer.text = 'Output layer'
   submodel2:add(outputLayer)

   -- Get output layer size
   table.insert(mapsizes, 1)
   table.insert(trainParam, nHiddenNeurons[#nHiddenNeurons] * nbClasses)
   table.insert(nConnections, trainParam[#trainParam])
   table.insert(nFeatureMaps, 1)
   table.insert(nHiddenNeurons, nbClasses)

   -- Computing memory usage
   local biasMem = 2 * nbClasses
   local weightMem = 2 * nHiddenNeurons[#nHiddenNeurons] * nbClasses
   local gradInputMem = opt.batchSize * nHiddenNeurons[#nHiddenNeurons]
   local outputMem = opt.batchSize * nbClasses
   table.insert(memory.submodel2.val, biasMem + weightMem + gradInputMem + outputMem)
   table.insert(memory.submodel2.str,'Lnr')

   if opt.probe then
      submodel1.val:add(nn.Probe('Probing output layer'))
   end
   --log probabilities
   local logsoft = nn.LogSoftMax()
   table.insert(memory.submodel2.val, 2 * outputMem)
   table.insert(memory.submodel2.str,'SM')

   --add submodels to model
   model:add(submodel1)
   model:add(submodel2)

   -- Loss: NLL
   local loss = nn.ClassNLLCriterion()

   if cuda then
      model:cuda()
   end

   -- Creates dummy file for plotting
   local pltStat = io.open('.pltStatData','w+')
   pltStat:write('#Layer\tParam\n')

   --print sizes of nConvLayers
   statFile:write('\n==> Network model\n')
   for i = 0, nConvLayers + #neuronsPerLinearLayer + 1 do

      if i == 0 then filterSize[0] = 0
      elseif filterSize[i] == nil then filterSize[i] = 1 end

      local s = string.format(
      '==> model layer %2d  -  filter size: %2d  |  spatial extent: %3dx%3d  |  feature maps: %3d  |  hidden neurons: %6d  |  parameters: %7d  |  connections: %9d\n',
      i, filterSize[i], mapsizes[i], mapsizes[i], nFeatureMaps[i], nHiddenNeurons[i], trainParam[i], nConnections[i]
      )
      if opt.verbose then io.write(s) end
      statFile:write(s)
      statFile:flush()

      -- Log model size and stat
      if i>0 and i<= nConvLayers then
         pltStat:write(string.format('Conv-%d\t%d\n',i,trainParam[i]))
      elseif i > nConvLayers then
         pltStat:write(string.format('MLP-%d\t%d\n',i-nConvLayers,trainParam[i]))
      end

   end

   io.close(pltStat)
   os.execute [[gnuplot -e "
   set term dumb 170 50;
   p './.pltStatData' u 0:2:xtic(1) w boxes t 'parameters'
   " | tee .pltStat]]

   -- Evaluate network's weight
   local w = model:getParameters()
   memory.parameters = w:size(1)
   local str = 'This is a ' .. memory.parameters .. '-parameters network\n'
   if opt.verbose then io.write(str) end
   statFile:write(str); statFile:flush()

   --print(model.modules)
   return model, logsoft, loss, dropout, memory

end

-- Useful functions for <get_model2> -------------------------------------------
-- Iterative function definition for recovering the dropout table
local function assignDropout(dropoutTable,module)
   if module.__typename == 'nn.Dropout' then
      table.insert(dropoutTable, module)
   end
end

local function recoverDropoutTable(dropoutTable,network)
   assignDropout(dropoutTable,network)
   if network.modules then
      for _,a in ipairs(network.modules) do
         recoverDropoutTable(dropoutTable,a)
      end
   end
end

-- Load model from file --------------------------------------------------------
function get_model2(networkFile, cuda)

   -- Load model from file
   local model = netToolkit.loadNet(networkFile)

   -- Recover the dropout pointers
   local dropout = {}
   recoverDropoutTable(dropout,model)

   -- Loss: NLL
   local logsoft = nn.LogSoftMax()
   local loss = nn.ClassNLLCriterion()

   if cuda then
      model:cuda()
   end

   return model, logsoft, loss, dropout

end

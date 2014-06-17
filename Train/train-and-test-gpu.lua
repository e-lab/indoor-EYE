-------------------------------------------------------------------------------
-- Functions for training and testing a neural network
-- Gregory Essertel, June 2014
-- based on the work of:
-- Artem Kuharenko
-- Alfredo Canziani, Feb 2014
-- Gregory Essertel, May 2014
-------------------------------------------------------------------------------

require 'optim'   -- an optimization package, for online and batch methods

local TopAcc = require('TopAccuracy')

--allocate memory for batch of images
local ims = torch.CudaTensor(opt.batchSize, 3, opt.side, opt.side)

--allocate memory for batch of labels
local targets = torch.Tensor(opt.batchSize)

-- w and dE_dw are global for this file
local w, dE_dw
local optimState

function train(datasetExtractor, model, logsoft, loss, dropout, top5)
   --train one iteration
   local nbThread = datasetExtractor:getNbThreads()
   datasetExtractor:newShuffle(true)

   for batch = 1, nbThread do
      datasetExtractor:prepareBatch(batch, true)
   end

   local ce_train_error = 0
   local loading_time = 0
   local computing_time = 0
   local nb_batches = datasetExtractor:getNbBatches(true)

   for batch = 1, nb_batches do

      xlua.progress(batch, nb_batches)

      --copy batch
      local timerLoading = torch.Timer()
      datasetExtractor:copyBatch(batch, ims, targets)
      loading_time = loading_time + timerLoading:time().real

      --prepare next batch
      if batch + nbThread <= nb_batches then
         datasetExtractor:prepareBatch(batch + nbThread, true)
      end

      -- create closure to evaluate f(X) and df/dX
      local eval_E = function (att)

         dE_dw:zero()

         local outputModelGPU = model:forward(ims)
         cutorch.synchronize()
         local outputModelCPU = outputModelGPU:float()
         local preds = logsoft:forward(outputModelCPU)
         local E = loss:forward(preds, targets)

         top5:batchAdd(preds, targets)
         ce_train_error = ce_train_error + E

         local dE_dy = loss:backward(preds, targets)
         local gradLogSoftCPU = logsoft:backward(outputModelCPU, dE_dy)

         -- on GPU
         local gradLogSoftGPU = gradLogSoftCPU:cuda()
         model:backward(ims, gradLogSoftGPU)
         cutorch.synchronize()

         return E, dE_dw
      end

      -- optimize on current mini-batch
      collectgarbage()
      local timerComputing = torch.Timer()
      optim.sgd(eval_E, w, optimState)
      computing_time = computing_time + timerComputing:time().real
   end

   loading_time = loading_time / nb_batches
   computing_time = computing_time / nb_batches
   ce_train_error = ce_train_error / nb_batches

   return ce_train_error, loading_time, computing_time
end

function test(datasetExtractor, model, logsoft, loss, dropout, top5)

   local ce_test_error = 0
   local loading_time = 0
   local computing_time = 0
   local nb_batches = datasetExtractor:getNbBatches(false)

   local nbThread = math.min(datasetExtractor:getNbThreads(), nb_batches)
   datasetExtractor:newShuffle(false)

   for batch = 1, nbThread do
      datasetExtractor:prepareBatch(batch, false)
   end

   -- Switching off the dropout
   if opt.dropout > 0 or opt.inputDO > 0 then
      for _,d in ipairs(dropout) do
         d.train = false
      end
   end

   for batch = 1, nb_batches do

      xlua.progress(batch, nb_batches)

      local timerLoading = torch.Timer()
      datasetExtractor:copyBatch(batch, ims, targets)
      loading_time = loading_time + timerLoading:time().real
      --prepare next batch
      if batch + nbThread <= nb_batches then
         datasetExtractor:prepareBatch(batch + nbThread, false)
      end

      -- test sample
      local timerComputing = torch.Timer()
      local outputModelGPU = model:forward(ims)
      cutorch.synchronize()
      local outputModelCPU = outputModelGPU:float()
      local preds = logsoft:forward(outputModelCPU)
      local E = loss:forward(preds, targets)

      top5:batchAdd(preds, targets)
      ce_test_error = ce_test_error + E

      computing_time = computing_time + timerComputing:time().real
   end

   -- Switching back on the dropout
   if opt.dropout > 0 or opt.inputDO > 0 then
      for _,d in ipairs(dropout) do
         d.train = true
      end
   end

   ce_test_error = ce_test_error / nb_batches
   loading_time = loading_time / nb_batches
   computing_time = computing_time / nb_batches

   return ce_test_error, loading_time, computing_time
end


function train_and_test(dataset, model, logsoft, loss, dropout, statFile, logger, ce_logger, logger_5)

   w, dE_dw = model:getParameters()

   --init confusion matricies
   local nb_classes = #dataset:getClasses()
   local train_top5 = TopAcc(nb_classes, 5)
   local test_top5 = TopAcc(nb_classes, 5)
   train_top5:zero()
   test_top5:zero()

   --set optimization parameters
   optimState = {
      learningRate = opt.learningRate,
      momentum = opt.momentum,
      weightDecay = opt.weightDecay,
      learningRateDecay = opt.learningRateDecay
   }

   local epoch = 1
   -- recover logger if restaring the training
   if opt.network ~= 'N/A' then
      -- (1) get the number of the epoch
      epoch = 1 + tonumber(string.match(opt.network, "%d+"))
   end

   local continue = true

   local total_timer = torch.Timer()
   while continue do
      collectgarbage()
      -------------------------------------------------------------------------------
      --train
      if opt.verbose then
         print('==> Train ' .. epoch)
      end
      local train_timer = torch.Timer()
      local train_ce_error, train_loading, train_computing = train(dataset, model, logsoft, loss, dropout, train_top5)
      local train_time = train_timer:time().real

      -- (2) testing
      if opt.verbose then
         print('==> Test ' .. epoch)
      end
      local test_timer = torch.Timer()
      local test_ce_error, test_loading, test_computing = test(dataset, model, logsoft, loss, dropout, test_top5)
      local test_time = test_timer:time().real

      -- (3) update loggers
      train_top5:update()
      test_top5:update()
      logger:add{['% train accuracy'] = train_top5.result[1] * 100, ['% test accuracy'] = test_top5.result[1] * 100}
      ce_logger:add{['ce train error'] = ce_train_error, ['ce test error'] = ce_test_error}
      logger_5:add{train_top5.result[5] * 100,test_top5.result[5] * 100,
      train_top5.result[1] * 100, test_top5.result[1] * 100}

      local comment = string.format("\n%s\n%s\n%s\n%s\n%s\n\n%s\n%s\n%s\n%s\n\n%s\n%s\n\n",
      string.format('-------------------- Statistics epoch %3d ---------------------', epoch),
      '\t\t\tTrain\t\tTest',
      string.format("Top5\t\t\t%.2f\t\t%.2f", train_top5.result[5] * 100, test_top5.result[5] * 100),
      string.format("Top1\t\t\t%.2f\t\t%.2f", train_top5.result[1] * 100, test_top5.result[1] * 100),
      string.format("CE error\t\t%.4f\t\t%.4f", train_ce_error, test_ce_error),
      string.format("Epoch(min)\t\t%.2f\t\t%.2f", train_time/60, test_time / 60),
      string.format("Batch Loading(msec)\t%.3f\t\t%.3f", train_loading * 1000, test_loading * 1000),
      string.format("Batch Computing(msec)\t%.3f\t\t%.3f", train_computing* 1000, test_computing * 1000),
      string.format("Batch Total(msec)\t%.3f\t\t%.3f", train_time/dataset:getNbBatches(true) * 1000, test_time /dataset:getNbBatches(false) * 1000),
      string.format("Total time(min)\t%.2f", total_timer:time().real/60),
      '---------------------------------------------------------------')

      if (opt.verbose) then
         print(comment)
      end

      train_top5:zero()
      test_top5:zero()

      if opt.plot then
         logger:plot()
         ce_logger:plot()
         logger_5:plot()
      end

      -- (4) save the network
      w = nil
      dE_dw = nil
      w, dE_dw = netToolkit.saveNet(model, opt.save_dir .. 'model-' .. epoch .. '.net', opt.verbose)

      -- (5) log in file
      statFile:write(comment)
      statFile:flush()

      -- (6) set up for the next epoch
      epoch = epoch + 1

      if (paths.filep('./.stop')) then
         os.execute 'rm .stop'
         continue = false
      end
   end
end

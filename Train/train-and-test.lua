-------------------------------------------------------------------------------
-- Functions for training and testing a classifier
-- Artem Kuharenko
-- Alfredo Canziani, Feb 2014
-------------------------------------------------------------------------------

require 'torch'   -- torch
require 'optim'   -- an optimization package, for online and batch methods
local TopAcc = require('TopAccuracy')

--allocate memory for batch of images
local ims = torch.Tensor(opt.batchSize, 3, opt.height, opt.width)
--allocate memory for batch of labels
local targets = torch.Tensor(opt.batchSize)

if opt.cuda then
   ims = ims:cuda()
   targets = targets:cuda()
end

local weightsBackup = {}
local topBackup = {}

local trainTestTime   = {}
local w, dE_dw

function train(data, model, loss, dropout, top5)
   --train one iteration

   local nbThread = opt.mm_threads
   data.newShuffle()

   for batch = 1, nbThread do
      data.prepareBatch(batch, false)
   end

   local t = 1
   local trainedSuccessfully = true
   local nbFailures = 0
   local consecutiveFailures = 0
   local olderBack = 1
   local last_ce_train_error = 0

   ce_train_error = 0

   weightsBackup[1]:copy(w)
   weightsBackup[2]:copy(w)
   local ceBackup = {0, 0}
   topBackup[1]:zero()
   topBackup[2]:zero()

   trainTestTime.tmpLoading = 0
   trainTestTime.tmpCuda = 0

   while t <= data.nBatches do

      xlua.progress(t, data.nBatches)

      --copy batch
      if (trainedSuccessfully) then
         if (t%10 == 0) then
            weightsBackup[olderBack]:copy(w)
            ceBackup[olderBack] = ce_train_error
            topBackup[olderBack].mat:copy(top5.mat)
            olderBack = 3 - olderBack
         end

         local timeB = sys.clock()
         data.copyBatch(t, ims, targets)

         --prepare next batch
         if t + nbThread <= data.nBatches then
            data.prepareBatch(t + nbThread, false)
         end
         trainTestTime.tmpLoading = trainTestTime.tmpLoading + (sys.clock() - timeB)
      else
         if (consecutiveFailures == 1) then
            w:copy(weightsBackup[3 - olderBack])
            top5.mat:copy(topBackup[3 - olderBack].mat)
            ce_train_error = ceBackup[3 - olderBack]
         else
            w:copy(weightsBackup[olderBack])
            ce_train_error = ceBackup[olderBack]
            top5.mat:copy(topBackup[olderBack].mat)
            weightsBackup[3 - olderBack]:copy(weightsBackup[olderBack])
            ceBackup[3 - olderBack] = ceBackup[olderBack]
            topBackup[3 - olderBack].mat:copy(topBackup[olderBack].mat)
         end
      end

      -- create closure to evaluate f(X) and df/dX
      local eval_E = function()

         dE_dw:zero()

         local y = model:forward(ims)
         local E = loss:forward(y, targets)

         top5:batchAdd(y, targets)

         ce_train_error = ce_train_error + E
         local dE_dy = loss:backward(y, targets)
         model:backward(ims, dE_dy)

         return E, dE_dw
      end

      -- optimize on current mini-batch
      collectgarbage()
      local timer = torch.Timer()
      optim.sgd(eval_E, w, optimState)
      trainTestTime.tmpCuda = trainTestTime.tmpCuda + timer:time().real

      if (ce_train_error < last_ce_train_error + 10) then
         trainedSuccessfully = true
         last_ce_train_error = ce_train_error
         consecutiveFailures = 0
         t = t + 1
      else
         trainedSuccessfully = false
         nbFailures = nbFailures + 1
         consecutiveFailures = consecutiveFailures + 1
         if (consecutiveFailures < 3) then
            print(sys.COLORS.red .. '\nWARNING:')
            print(sys.COLORS.red .. 'Failed training on current batch: ' .. ce_train_error .. '. Try again with backup parameters.')
         else
            -- stop the loop --
            t = data.nBatches + 1
         end
      end
   end

   if (consecutiveFailures < 3) then
      ce_train_error = ce_train_error / data.nBatches
      trainTestTime.tmpLoading = trainTestTime.tmpLoading / data.nBatches
      trainTestTime.tmpCuda = trainTestTime.tmpCuda / data.nBatches

      return true, nbFailures
   else
      print(sys.COLORS.red .. '\nFailed training 3 times in a row')
      print(sys.COLORS.red .. 'Total failures:' .. nbFailures)

      return false, nbFailures
   end
end

function test(data, model, loss, dropout, top5)

   local nbThread = opt.mm_threads
   data.newShuffle()

   for batch = 1, math.min(nbThread, data.nBatches) do
      data.prepareBatch(batch, true)
   end

   -- Switching off the dropout
   if opt.dropout > 0 or opt.inputDO > 0 then
      for _,d in ipairs(dropout) do
         d.train = false
      end
   end

   for t = 1, data.nBatches do

      xlua.progress(t, data.nBatches)

      data.copyBatch(t, ims, targets)
      --prepare next batch
      if t + nbThread <= data.nBatches then
         data.prepareBatch(t + nbThread, true)
      end

      -- test sample
      local preds = model:forward(ims)
      local E = loss:forward(preds, targets)

      top5:batchAdd(preds, targets)
      ce_test_error = ce_test_error + E
   end

   -- Switching back on the dropout
   if opt.dropout > 0 or opt.inputDO > 0 then
      for _,d in ipairs(dropout) do
         d.train = true
      end
   end

   ce_test_error = ce_test_error / data.nBatches

end

function checkWeight(model, logMin, logMax, logAvg, logStd, logGwsMin, logGwsMax, logGwsAvg, logGwsStd)
   local NaNOk = true
   --print weights and gradweights statistics
   if opt.print_weight_stat or opt.debug then

      local wsMin,  wsMax,  wsAvg,  wsStd  = {},{},{},{}
      local gwsMin, gwsMax, gwsAvg, gwsStd = {},{},{},{}
      local style = {}

      for _,seq in ipairs(model.modules) do
         for _,m in ipairs(seq.modules) do
            if m.printable then

               -- Computing <weight> statistics
               local ws = m.weight:float()
               if opt.debug then
                  -- Detecting and removing NaNs
                  if ws:ne(ws):sum() > 0 then
                     print(sys.COLORS.red .. m.text .. ' weights has NaN/s')
                     NaNOk = false
                  end
                  ws[ws:ne(ws)] = 0

                  wsMin[m.text] = ws:min()
                  wsMax[m.text] = ws:max()
                  wsAvg[m.text] = ws:mean()
                  wsStd[m.text] = ws:std()
               end

               -- Compute max L2 norw of neuron weights
               local maxL2 = 0
               for i2 = 1, ws:size(1) do
                  local neuronL2 = ws[i2]:norm()
                  if neuronL2 > maxL2 then
                     maxL2 = neuronL2
                  end
               end

               ws = ws:abs()
               local ws_small = ws:lt(1e-5):sum()
               local ws_big = ws:gt(1e+2):sum()

               -- Computing <gradients> statistics
               local gws = m.gradWeight:float()
               if opt.debug then
                  -- Detecting and removing NaNs
                  if gws:ne(gws):sum() > 0 then
                     print(sys.COLORS.red .. m.text .. ' gradients has NaN/s')
                     NaNOk = false
                  end
                  gws[gws:ne(gws)] = 0

                  gwsMin[m.text] = gws:min()
                  gwsMax[m.text] = gws:max()
                  gwsAvg[m.text] = gws:mean()
                  gwsStd[m.text] = gws:std()
               end

               gws = gws:abs()
               local gws_small = gws:lt(1e-5):sum()
               local gws_big = gws:gt(1e+2):sum()

               -- Setting plotting style
               if opt.debug then style[m.text] = '-' end

               -- Printing some stats
               if opt.print_weight_stat then
                  print(m.text)
                  print(string.format('max L2 weights norm: %f', maxL2))
                  print(string.format('#small weights: %d, big weights: %d', ws_small, ws_big))
                  print(string.format('#small grads  : %d, big grads  : %d', gws_small, gws_big))
               end

            end
         end
      end

      -- Logging stats
      if opt.debug then
         logWsMin :add(wsMin )
         logWsMax :add(wsMax )
         logWsAvg :add(wsAvg )
         logWsStd :add(wsStd )
         logGwsMin:add(gwsMin)
         logGwsMax:add(gwsMax)
         logGwsAvg:add(gwsAvg)
         logGwsStd:add(gwsStd)
      end

      -- Plotting
      if plot and opt.debug then
         -- Setting the style
         logWsMin :style(style)
         logWsMax :style(style)
         logWsAvg :style(style)
         logWsStd :style(style)
         logGwsMin:style(style)
         logGwsMax:style(style)
         logGwsAvg:style(style)
         logGwsStd:style(style)

         -- Plotting
         logWsMin :plot()
         logWsMax :plot()
         logWsAvg :plot()
         logWsStd :plot()
         logGwsMin:plot()
         logGwsMax:plot()
         logGwsAvg:plot()
         logGwsStd:plot()
      end

   end

   if (opt.verbose and not NaNOk) then
      print()
      print(sys.COLORS.red .. '>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<')
      print(sys.COLORS.red .. '>>> NaN detected! Retraining same epoch! <<<')
      print(sys.COLORS.red .. '>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<')
      print('\n')
   end

   return NaNOk
end

function train_and_test(trainData, testData, model, loss, plot, verbose, dropout)

   w, dE_dw = model:getParameters()

   --init confusion matricies
   local train_top5 = TopAcc(#classes, 5)
   local test_top5 = TopAcc(#classes, 5)
   train_top5:zero()
   test_top5:zero()

   topBackup[1] = TopAcc(#classes, 5)
   topBackup[2] = TopAcc(#classes, 5)

   --set optimization parameters
   optimState = {
      learningRate = opt.learningRate,
      momentum = opt.momentum,
      weightDecay = opt.weightDecay,
      learningRateDecay = opt.learningRateDecay
   }

   --init logger for train and test accuracy
   local logger
   --init logger for train and test cross-entropy error
   local ce_logger
   --init logger for train and test accuracy-5
   local logger_5

   local epochInit = 0
   local epoch = 1
   local prevTestAcc = 0
   if opt.network ~= 'N/A' then
      -- (1) get the number of the epoch
      epochInit = tonumber(string.match(opt.network, "%d+"))
      epoch = epochInit + 1

      -- (2) create tmp tables
      local tmpAcc = {}
      local tmpCE = {}
      local tmpAcc5 = {}
      if (paths.filep(opt.save_dir .. 'accuracy.log')) then
         local file = io.open(opt.save_dir .. 'accuracy.log', r)
         local tmpVal = file:read("*line")  -- skip the header
         for ep = 1, epochInit do
            tmpVal = file:read("*number")  -- train
            table.insert(tmpAcc, tmpVal)
            tmpVal = file:read("*number")  -- test
            table.insert(tmpAcc, tmpVal)
         end
         io.close(file)
      end
      if (paths.filep(opt.save_dir .. 'cross-entropy.log')) then
         local file = io.open(opt.save_dir .. 'cross-entropy.log', r)
         local tmpVal = file:read("*line")  -- skip the header
         for ep = 1, epochInit do
            tmpVal = file:read("*number")  -- train
            table.insert(tmpCE, tmpVal)
            tmpVal = file:read("*number")  -- test
            table.insert(tmpCE, tmpVal)
         end
         io.close(file)
      end
      if (paths.filep(opt.save_dir .. 'accuracy-5.log')) then
         local file = io.open(opt.save_dir .. 'accuracy-5.log', r)
         local tmpVal = file:read("*line")  -- skip the header
         for ep = 1, epochInit do
            for log = 1, 3 do
               tmpVal = file:read("*number")  -- train
               table.insert(tmpAcc5, tmpVal)
               tmpVal = file:read("*number")  -- test
               table.insert(tmpAcc5, tmpVal)
            end
         end
         io.close(file)
      end

      -- (3) init loggers
      logger = optim.Logger(opt.save_dir .. 'accuracy.log')
      ce_logger = optim.Logger(opt.save_dir .. 'cross-entropy.log')
      logger_5 = optim.Logger(opt.save_dir .. 'accuracy-5.log')
      logger_5:setNames({'% train top1', '% test top1',
      '% train top3', '% test top3',
      '% train top5', '% test top5'})

      -- (4) load backup data
      for ep = 1, epochInit do
         logger:add{['% train accuracy'] = tmpAcc[2*ep-1], ['% test accuracy'] = tmpAcc[2*ep]}
         ce_logger:add{['ce train error'] = tmpCE[2*ep-1], ['ce test error'] = tmpCE[2*ep]}
         logger_5:add{tmpAcc5[6*ep-5], tmpAcc5[6*ep-4],
         tmpAcc5[6*ep-3], tmpAcc5[6*ep-2],
         tmpAcc5[6*ep-1], tmpAcc5[6*ep]}
      end

      -- (5) plot
      if plot then
         logger:style{['% train accuracy'] = '-', ['% test accuracy'] = '-'}
         logger:plot()
         ce_logger:style{['ce train error'] = '-', ['ce test error'] = '-'}
         ce_logger:plot()
         logger_5:style{'-', '-', '-', '-', '-', '-'}
         logger_5:plot()
      end

      -- (6) backup previous accuracy
      prevTestAcc = tmpAcc[#tmpAcc]/100
   else
      logger = optim.Logger(opt.save_dir .. 'accuracy.log')
      ce_logger = optim.Logger(opt.save_dir .. 'cross-entropy.log')
      logger_5 = optim.Logger(opt.save_dir .. 'accuracy-5.log')
      logger_5:setNames({'% train top1', '% test top1',
      '% train top3', '% test top3',
      '% train top5', '% test top5'})
   end

   --init train and test time
   trainTestTime.train   = {}
   trainTestTime.test    = {}

   for _,t in pairs(trainTestTime) do
      t.perSample = 0
      t.total     = 0
   end

   trainTestTime.loading = 0
   trainTestTime.cuda = 0
   trainTestTime.tmpLoading = 0
   trainTestTime.tmpCuda = 0

   -- Initialising debugging loggers
   local logMin, logMax, logAvg, logStd, logGwsMin, logGwsMax, logGwsAvg, logGwsStd
   if opt.debug then
      logWsMin  = optim.Logger(opt.save_dir .. 'logWsMin.log' )
      logWsMax  = optim.Logger(opt.save_dir .. 'logWsMax.log' )
      logWsAvg  = optim.Logger(opt.save_dir .. 'logWsAvg.log' )
      logWsStd  = optim.Logger(opt.save_dir .. 'logWsStd.log' )
      logGwsMin = optim.Logger(opt.save_dir .. 'logGwsMin.log')
      logGwsMax = optim.Logger(opt.save_dir .. 'logGwsMax.log')
      logGwsAvg = optim.Logger(opt.save_dir .. 'logGwsAvg.log')
      logGwsStd = optim.Logger(opt.save_dir .. 'logGwsStd.log')
   end

   weightsBackup = {w:clone(), w:clone()}

   local trainedSuccessfully = false
   local hasNaN
   local weightsBackup = w:clone()
   local nbFailures = 0
   local continue = true
   local epochTrained = 0

   while continue do
      collectgarbage()
      -------------------------------------------------------------------------------
      --train
      sys.tic()
      if verbose then print('==> Train ' .. epoch) end
      trainedSuccessfully, nbFailures = train(trainData, model, loss, dropout, train_top5)
      local time = sys.toc()

      -------------------------------------------------------------------------------
      -- check validity
      -- (1) Nan in weight
      if (trainedSuccessfully) then
         trainedSuccessfully = checkWeight(model, logMin, logMax, logAvg, logStd, logGwsMin, logGwsMax, logGwsAvg, logGwsStd)
      end

      -- (2) testing and checking if there is a drop in the accuracy
      local timeTest = 0
      if (trainedSuccessfully) then
         sys.tic()
         ce_test_error = 0
         if verbose then print('==> Test ' .. epoch) end
         test(testData, model, loss, dropout, test_top5)
         timeTest = sys.toc()
         test_top5:update()

         trainedSuccessfully = test_top5.result[1] >  0.5 * prevTestAcc
      end

      -- (3) if every thing is good the procced
      if (trainedSuccessfully) then

         -- (1) backup weight
         print(sys.COLORS.green .. '==> Epoch trained successfully - backing up network\'s weights')
         weightsBackup:copy(w)

         -- (2) Statistics
         trainTestTime.train.perSample = trainTestTime.train.perSample + time / (opt.batchSize * trainData.nBatches)
         trainTestTime.train.total     = trainTestTime.train.total + time
         trainTestTime.loading = trainTestTime.loading + trainTestTime.tmpLoading
         trainTestTime.cuda = trainTestTime.cuda + trainTestTime.tmpCuda
         epochTrained = epochTrained + 1

         if verbose then
            print(string.format("======> Time to learn 1 iteration = %.2f sec", time))
            print(string.format("======> Time to train 1 sample = %.2f ms", time / (opt.batchSize * trainData.nBatches) * 1000))
            print(string.format("======> Train CE error: %.2f", ce_train_error))
            print(string.format("======> Time to load 1 batch = %.2f msec", trainTestTime.tmpLoading * 1000))
            print(string.format("======> Time to cumpute 1 batch on the GPU = %.2f msec", trainTestTime.tmpCuda * 1000))
            if nbFailures > 0 then
               print(sys.COLORS.red .. "======> Number of failures:" ..  nbFailures)
            end
            train_top5:update()
            print(string.format('======> Train accuray %.3f', train_top5.result[1]*100))
            print()
         end

         -- (2.2) testing statistics
         trainTestTime.test.perSample = trainTestTime.test.perSample + timeTest / (opt.batchSize * trainData.nBatches)
         trainTestTime.test.total     = trainTestTime.test.total + timeTest

         if verbose then
            print(string.format("======> Time to test 1 iteration = %.2f sec", timeTest))
            print(string.format("======> Time to test 1 sample = %.2f ms", timeTest / (opt.batchSize * testData.nBatches) *  1000))
            print(string.format("======> Test CE error: %.2f", ce_test_error))
            test_top5:update()
            print(string.format('======> Test accuray %.3f', test_top5.result[1]*100))
            print()
         end

         -- (5) update loggers
         logger:add{['% train accuracy'] = train_top5.result[1] * 100, ['% test accuracy'] = test_top5.result[1] * 100}
         ce_logger:add{['ce train error'] = ce_train_error, ['ce test error'] = ce_test_error}
         logger_5:add{train_top5.result[1] * 100,test_top5.result[1] * 100,
         train_top5.result[3] * 100, test_top5.result[3] * 100,
         train_top5.result[5] * 100, test_top5.result[5] * 100}

         --plot
         if plot then
            logger:style{['% train accuracy'] = '-', ['% test accuracy'] = '-'}
            logger:plot()
            ce_logger:style{['ce train error'] = '-', ['ce test error'] = '-'}
            ce_logger:plot()
            logger_5:style{'-', '-', '-', '-', '-', '-'}
            logger_5:plot()
         end

         -- (6) save the network
         w = nil
         dE_dw = nil
         w, dE_dw = netToolkit.saveNet(opt.save_dir .. 'model-' .. epoch .. '.net', model)

         -- (7) log in file
         if (epoch % 5 == 0) then
            statFile:write(string.format('\nTraining & testing time for %d epochs: %.2f minutes\n', epoch - epochInit, (trainTestTime.train.total + trainTestTime.test.total)/60))
            statFile:write(string.format('Average training time per sample: %.3f ms\n', trainTestTime.train.perSample * 1000 / epochTrained))
            statFile:write(string.format('Average testing time per sample: %.3f ms\n', trainTestTime.test.perSample * 1000 / epochTrained))
            statFile:write(string.format('Average loading time per batch: %.3f ms\n', trainTestTime.loading * 1000 / epochTrained))
            statFile:write(string.format('Average cuda time per batch: %.3f ms\n', trainTestTime.cuda * 1000 / epochTrained))

            trainTestTime.train.perSample = 0
            trainTestTime.test.perSample = 0
            trainTestTime.loading = 0
            trainTestTime.cuda = 0
            epochTrained = 0

            statFile:flush()
         end

         -- Save last accuracy before zeroing
         prevTestAcc = test_top5.result[1]
         train_top5:zero()
         test_top5:zero()

         -- (7) set up for the next epoch
         epoch = epoch + 1
      else -- (trainsucessfully) -- start on same epoch
         print()
         print(sys.COLORS.red .. '>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<')
         print(sys.COLORS.red .. '>>>> Drop in testing > 50% <<<<')
         print(sys.COLORS.red .. '>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<')
         print(test_top5.result[1] ,'\n')
         -- (1) reset weights
         w:copy(weightsBackup)
      end

      if (paths.filep('./.stop')) then
         print(sys.COLORS.blue .. 'Stop flag detected. Exiting cleanly.')
         os.execute 'rm .stop'
         continue = false
      end
   end

   -------------------------------------------------------------------------------

   --compute train and test accuracy. Average over last 5 iterations
   --[[local test_accs = logger.symbols['% test accuracy']
   local train_accs = logger.symbols['% train accuracy']
   local test_acc = 0
   local train_acc = 0

   for i = 0, 4 do
      test_acc = test_acc + test_accs[#test_accs - i]
      train_acc = train_acc + train_accs[#train_accs - i]
   end

   test_acc = test_acc / 5
   train_acc = train_acc / 5

   --compute train and test cross entropy error. Average over last 5 iterations
   local test_errs = ce_logger.symbols['ce test error']
   local train_errs = ce_logger.symbols['ce train error']
   local test_err = 0
   local train_err = 0

   for i = 0, 4 do
      test_err = test_err + test_errs[#test_errs - i]
      train_err = train_err + train_errs[#train_errs - i]
   end

   test_err = test_err / 5
   train_err = train_err / 5

   -- Output statistics ----------------------------------------------------------
   str = {}
   -- printing average timing
   str[ 1] = string.format('\n\n')
   str[ 2] = string.format('==> Global statistics\n')
   str[ 3] = string.format('==> Timing\n')
   -- Total time
   str[ 4] = string.format('    Total time for training the network: %.2f minutes\n', trainTestTime.train.total/60)
   str[ 5] = string.format('    Total time for testing the network: %.2f minutes\n', trainTestTime.test.total/60)
   str[ 6] = string.format('    Total time: %.2f minutes\n', (trainTestTime.train.total + trainTestTime.test.total)/60)
   -- Per sample time
   str[ 7] = string.format('    Average training time per sample: %.3f ms\n', trainTestTime.train.total * 1000)
   str[ 8] = string.format('    Average testing time per sample: %.3f ms\n', trainTestTime.test.perSample * 1000)
   -- Performance
   str[ 9] = string.format('==> Performance\n')
   str[10] = string.format('    Train accuracy = %.3f%%\n', train_acc)
   str[11] = string.format('    Test accuracy = %.3f%%\n', test_acc)
   str[12] = string.format('    Train cross-entropy error = %.3f\n', train_err)
   str[13] = string.format('    Test cross-entropy error = %.3f\n', test_err)

   -- Printing on screen
   for _,s in ipairs(str) do io.write(s) end

   -- Logging data on file
   for _,s in ipairs(str) do statFile:write(s) end
   statFile:close()
   --]]
   return train_acc, test_acc
end

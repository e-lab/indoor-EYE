-------------------------------------------------------------------------------
-- Functions for training and testing a classifier
-- Artem Kuharenko
-- Alfredo Canziani, Feb 2014
-------------------------------------------------------------------------------

require 'torch'   -- torch
require 'optim'   -- an optimization package, for online and batch methods

function train(data, model, loss, dropout)
   --train one iteration

   data.prepareBatch(1)

   for t = 1, data.nbatches() do

      xlua.progress(t, data.nbatches())

      --      local t0 = sys.clock()
      --copy batch
      ims, targets = data.copyBatch()

      --prepare next batch
      if t < data.nbatches() then
         data.prepareBatch(t + 1)
      end
      --      print('load data: ' .. sys.clock() - t0)

      -- create closure to evaluate f(X) and df/dX
      local eval_E = function(w)

         dE_dw:zero()

         local y = model:forward(ims)
         local E = loss:forward(y, targets)
         ce_train_error = ce_train_error + E * opt.batchSize

         local dE_dy = loss:backward(y, targets)
         model:backward(ims, dE_dy)

         return E, dE_dw

      end

      -- optimize on current mini-batch
      optim.sgd(eval_E, w, optimState)

      -- Switching off the dropout
      if opt.dropout > 0 or opt.inputDO > 0 then
         for _,d in ipairs(dropout) do
            d.train = false
         end
      end

      -- Update confusion matrix
      local y = model:forward(ims)
      for i = 1, opt.batchSize do
         train_confusion:add(y[i], targets[i])
      end

      -- Switching back on the dropout
      if opt.dropout > 0 or opt.inputDO > 0 then
         for _,d in ipairs(dropout) do
            d.train = true
         end
      end

   end

   ce_train_error = ce_train_error / (data.nbatches() * opt.batchSize)

end

function test(data, model, loss, dropout)

   -- Switching off the dropout
   if opt.dropout > 0 or opt.inputDO > 0 then
      for _,d in ipairs(dropout) do
         d.train = false
      end
   end

   for t = 1, data.nbatches() do

      xlua.progress(t, data.nbatches())
      data.prepareBatch(t)
      ims, targets = data.copyBatch()

      -- test sample
      local preds = model:forward(ims)

      -- confusion
      for i = 1, opt.batchSize do

         test_confusion:add(preds[i], targets[i])
         local E = loss:forward(preds[i], targets[i])
         ce_test_error = ce_test_error + E

      end

   end

   -- Switching back on the dropout
   if opt.dropout > 0 or opt.inputDO > 0 then
      for _,d in ipairs(dropout) do
         d.train = true
      end
   end

   ce_test_error = ce_test_error / (data.nbatches() * opt.batchSize)

end

--reduced version of optim.ConfusionMatrix:__tostring__
function print_confusion_matrix(self, init_str)

   self:updateValids()
   local str = {init_str}

   if opt.print_confusion_matrix then

      str = {init_str .. ' ConfusionMatrix:\n'}
      local nclasses = self.nclasses
      table.insert(str, '[')
      for t = 1,nclasses do
         local pclass = self.valids[t] * 100
         pclass = string.format('%6.3f', pclass)
         if t == 1 then
            table.insert(str, '[')
         else
            table.insert(str, ' [')
         end
         for p = 1,nclasses do
            table.insert(str, string.format('%' .. opt.confusion_matrix_tab .. 'd', self.mat[t][p]))
         end
         if self.classes and self.classes[1] then
            if t == nclasses then
               table.insert(str, ']]  ' .. pclass .. '%  [class: ' .. (self.classes[t] or '') .. ']\n')
            else
               table.insert(str, ']   ' .. pclass .. '%  [class: ' .. (self.classes[t] or '') .. ']\n')
            end
         else
            if t == nclasses then
               table.insert(str, ']] ' .. pclass .. '% \n')
            else
               table.insert(str, ']   ' .. pclass .. '% \n')
            end
         end
      end


   end

   table.insert(str, string.format(' accuracy: %.3f', self.totalValid*100) .. '%')
   print(table.concat(str))

end

function train_and_test(trainData, testData, model, loss, plot, verbose, dropout)

   if verbose then
      print '==> training neuralnet'
   end
   w, dE_dw = model:getParameters()

   --init confusion matricies
   train_confusion = optim.ConfusionMatrix(classes)
   test_confusion = optim.ConfusionMatrix(classes)
   train_confusion:zero()
   test_confusion:zero()

   --set optimization parameters
   optimState = {
      learningRate = opt.learningRate,
      momentum = opt.momentum,
      weightDecay = opt.weightDecay,
      learningRateDecay = opt.learningRateDecay
   }

   --get image size
   local ivch = trainData.data:size(2)
   local ivhe = trainData.data:size(3)
   local ivwi = trainData.data:size(4)

   --init logger for train and test accuracy
   local logger = optim.Logger(opt.save_dir .. 'logger.log')
   --init logger for train and test cross-entropy error
   local ce_logger = optim.Logger(opt.save_dir .. 'celogger.log')
   --init train and test time
   local trainTestTime = {}
   trainTestTime.train = {}
   trainTestTime.test  = {}
   for _,t in pairs(trainTestTime) do
      t.perSample = torch.Tensor(opt.niters)
      t.total     = torch.zeros(opt.niters)
   end

   --allocate memory for batch of images
   ims = torch.Tensor(opt.batchSize, ivch, ivhe, ivwi)
   --allocate memory for batch of labels
   targets = torch.Tensor(opt.batchSize)

   if opt.type == 'cuda' then
      ims = ims:cuda()
      targets = targets:cuda()
   end

   for i = 1, opt.niters do

      -------------------------------------------------------------------------------
      --train
      local time = sys.clock()

      ce_train_error = 0
      if verbose then print('==> Train ' .. i) end
      train(trainData, model, loss, dropout)

      time = sys.clock() - time
      trainTestTime.train.perSample[i] = time / trainData.data:size(1)
      trainTestTime.train.total    [i] = time

      if verbose then
         print(string.format("======> Time to learn 1 iteration = %.2f sec", time))
         print(string.format("======> Time to train 1 sample = %.2f ms", time / trainData.data:size(1) * 1000))
         print(string.format("======> Train CE error: %.2f", ce_train_error))

         print_confusion_matrix(train_confusion, '======> Train')

         --print weights and gradweights statistics
         if opt.print_weight_stat then

            for _,seq in ipairs(model.modules) do
               for _,m in ipairs(seq.modules) do
                  if m.printable then

                     local ws = m.weight:clone() --weights
                     ws = ws:float():abs()
                     local ws_small = ws:lt(1e-5):sum()
                     local ws_big = ws:gt(1e+2):sum()

                     local gws = m.gradWeight:clone() --gradweights
                     gws = gws:float():abs()
                     local gws_small = gws:lt(1e-5):sum()
                     local gws_big = gws:gt(1e+2):sum()

                     print(m.text .. string.format(': number of small weights: %d, big weights: %d', ws_small, ws_big))
                     print(m.text .. string.format(': number of small gradweights: %d, big gradweights: %d', gws_small, gws_big))

                  end
               end
            end


         end

         print('\n')

      else
         xlua.progress(i, opt.niters)
      end
      -------------------------------------------------------------------------------

      -------------------------------------------------------------------------------
      --test
      local time = sys.clock()
      ce_test_error = 0
      if verbose then print('==> Test ' .. i) end
      test(testData, model, loss, dropout)
      time = sys.clock() - time
      trainTestTime.test.perSample[i] = time / testData.data:size(1)
      trainTestTime.test.total    [i] = time

      if verbose then
         print(string.format("======> Time to test 1 iteration = %.2f sec", time))
         print(string.format("======> Time to test 1 sample = %.2f ms", time / testData.data:size(1) * 1000))
         print(string.format("======> Test CE error: %.2f", ce_test_error))
         print_confusion_matrix(test_confusion, '======> Test')
         print()
      end
      -------------------------------------------------------------------------------

      -- update loggers
      train_confusion:updateValids()
      test_confusion:updateValids()
      logger:add{['% train accuracy'] = train_confusion.totalValid * 100, ['% test accuracy'] = test_confusion.totalValid * 100}
      ce_logger:add{['ce train error'] = ce_train_error, ['ce test error'] = ce_test_error}

      --plot
      if plot then
         logger:style{['% train accuracy'] = '-', ['% test accuracy'] = '-'}
         logger:plot()
         ce_logger:style{['ce train error'] = '-', ['ce test error'] = '-'}
         ce_logger:plot()
      end

      --save model every 5 iterations
      if (i % 5 == 0) then
         saveNet(model, opt.save_dir .. 'model-' .. i .. '.net', verbose)
         statFile:write(string.format('\nTraining & testing time for %d epochs: %.2f minutes\n', i, (trainTestTime.train.total:sum() + trainTestTime.test.total:sum())/60))
         statFile:write(string.format('Average training time per sample: %.3f ms\n', trainTestTime.train.perSample[{ {i-4,i} }]:mean() * 1000))
         statFile:write(string.format('Average testing time per sample: %.3f ms\n', trainTestTime.test.perSample[{ {i-4,i} }]:mean() * 1000))
         statFile:flush()
      end

      train_confusion:zero()
      test_confusion:zero()

   end

   -------------------------------------------------------------------------------

   --compute train and test accuracy. Average over last 5 iterations
   local test_accs = logger.symbols['% test accuracy']
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
   str[ 4] = string.format('    Total time for training the network: %.2f minutes\n', trainTestTime.train.total:sum()/60)
   str[ 5] = string.format('    Total time for testing the network: %.2f minutes\n', trainTestTime.test.total:sum()/60)
   str[ 6] = string.format('    Total time: %.2f minutes\n', (trainTestTime.train.total:sum() + trainTestTime.test.total:sum())/60)
   -- Per sample time
   str[ 7] = string.format('    Average training time per sample: %.3f ms\n', trainTestTime.train.perSample:mean() * 1000)
   str[ 8] = string.format('    Average testing time per sample: %.3f ms\n', trainTestTime.test.perSample:mean() * 1000)
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

   return train_acc, test_acc

end

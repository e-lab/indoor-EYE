#!/usr/bin/env torch
-------------------------------------------------------------------------------
-- Main script for training neural network
-- Artem Kuharenko
-- Alfredo Canziani, Feb 2014
-------------------------------------------------------------------------------

-- Require packages -----------------------------------------------------------
require 'torch'
require 'nn'
require 'torchffi'
require 'pl'
require 'usefulFunctions'

-- Options --------------------------------------------------------------------
opt = lapp[[
********************************************************************************
>>>>>>>>>>>>>>>>>>>>> indoor-NET: training on imageNet <<<<<<<<<<<<<<<<<<<<<<<<<
********************************************************************************

Dataset's parameters
   --side    (default 46  ) Training and testing image's side length (max 256)
   --colour  (default true) True by default, allows to train on B&W if the flag is used
   --jitter  (default 4   )

Learning parameters
   --learningRate      (default 5e-3)
   --learningRateDecay (default 1e-7)
   --weightDecay       (default 0   )
   --momentum          (default 0   )
   --batchSize         (default 32  )
   --niters            (default 20  )

Saving parameters
Be carefull!!! you need to save data when you change data options like width
   --save_dir        (default './results/'               )
   --temp_dir        (default './temp-data/'             )
   --train_data_file (default 'train-indoor-info-1300.t7')
   --train_info_file (default 'train-indoor-info-1300.t7')
   --test_data_file  (default 'test-indoor-data-50.t7'   )
   --test_info_file  (default 'test-indoor-info-50.t7'   )
   --data_sl         (default 'load'                     ) save once and then load prepared data from temp file ([load]|save)
   --mean_sl         (default 'load'                     ) save once and then load prepared mean from temp file ([load]|save)

On screen output
   --plot
   --verbose                (default true )
   --confusion_matrix_tab   (default 3    ) number of tabs between numbers in confusion matrix
   --print_confusion_matrix
   --print_weight_stat      (default true ) print number of neuralnet weights which are lower 1e-5 and higher 1e+2

CUDA parameters
   --cuda  (default true)
   --devid (default 1   ) device ID (if using CUDA)

Other parameters
   --verify_statistics
   --save              (default true )
   --mmload                            memory mapping when loading data. Use with small RAM
   --seed              (default 123  )
   --num_threads       (default 3    )
]]

-- Title ----------------------------------------------------------------------
if opt.verbose then print [[
********************************************************************************
>>>>>>>>>>>>>>>>>>>>> indoor-NET: training on imageNet <<<<<<<<<<<<<<<<<<<<<<<<<
********************************************************************************
]]

-- Aggiusting options ---------------------------------------------------------
   opt.width  = opt.side
   opt.height = opt.side
   opt.side   = nil

-- Print options summary ------------------------------------------------------
   print('==> Options:')
   for a,b in pairs(opt) do print('     + ' .. a .. ':', b) end
   print()
end

-- Aggiusting options ---------------------------------------------------------
if opt.colour then opt.ncolors = 3 else opt.ncolors = 1 end

-- Training with GPU (if CUDA) ------------------------------------------------
if opt.cuda then
   if opt.verbose then
      print('==> switching to CUDA')
   end
   require 'cutorch'
   require 'cunn'
   cutorch.setDevice(opt.devid)
   if opt.verbose then
      print('==> using GPU #' .. cutorch.getDevice())
      print(cutorch.getDeviceProperties(opt.devid))
   end
end

-- Loading functions ----------------------------------------------------------
if opt.verbose then print('==> Loading <training> and <testing> functions') end
dofile('train-and-test.lua')
if opt.verbose then print('==> Loading <model> functions') end
dofile('models.lua')
if opt.verbose then print('==> Loading <data> functions') end
dofile('Data/data-imagenet.lua') --imagenet data scripts
dofile('Data/data-process.lua') --data scripts

os.execute('mkdir -p ' .. opt.temp_dir) --create folder for temporary data
-------------------------------------------------------------------------------

data_folder = eex.datasetsPath() .. 'imagenet2012/'
train_data_file = data_folder .. opt.train_data_file
train_info_file = data_folder .. opt.train_info_file
test_data_file = data_folder .. opt.test_data_file
test_info_file = data_folder .. opt.test_info_file

dofile('Data/indoor-classes.lua')
classes = {}
for i = 1, #indoor_classes do classes[i] = indoor_classes[i][1] end

-------------------------------------------------------------------------------

torch.manualSeed(opt.seed)
torch.setnumthreads(opt.num_threads) --some of data scripts may change numthreads, so we need to do it here
torch.setdefaulttensortype('torch.FloatTensor')
os.execute('mkdir -p ' .. opt.save_dir) --create folder for saving results

--compute global mean and std
global_mean, global_std = get_global_mean_async(train_data_file, train_info_file, opt.save_dir, opt.mean_sl, opt.verbose)

--prepare data
if opt.mmload then
   testData = prepare_async(test_data_file, test_info_file)
   trainData = prepare_async(train_data_file, train_info_file)
else
   testData = prepare_sync(test_data_file, test_info_file, 'test.t7', opt.data_sl)
   trainData = prepare_sync(train_data_file, train_info_file, 'train.t7', opt.data_sl)
end
torch.manualSeed(opt.seed)

--show objects of different classes
--show_classes(trainData, 200, classes)

function run(trainData, testData)
--main function

   --print train and test image sizes
   print_sizes(trainData, 'Train', opt.verbose)
   print_sizes(testData, 'Test', opt.verbose)

   --print train and test mean and std
   verify_statistics(trainData.data, {'r','g','b'}, 'train images', opt.verify_statistics)
   verify_statistics(testData.data, {'r','g','b'}, 'test images', opt.verify_statistics)

   --get classifier and loss function
   local model, loss = get_model1() --(classifier.lua)

   --train classifier
   local train_acc, test_acc = train_and_test(trainData, testData, model, loss, opt.plot, opt.verbose) --(train-and-test.lua)
   -------------------------------------------------------------------------------

   --return train and test accuracy
   return train_acc, test_acc

end

train_acc, test_acc = run(trainData, testData)
print(string.format('==> Train accuracy = %.3f', train_acc) .. '%')
print(string.format('==> Test accuracy = %.3f', test_acc) .. '%')


#!/usr/bin/env torch
-------------------------------------------------------------------------------
-- Main script for training neural network
-- Artem Kuharenko
-- Alfredo Canziani, Feb 2014
-------------------------------------------------------------------------------

-- Require packages -----------------------------------------------------------
require 'torch'
require 'nnx'
require 'net-toolkit'
ffi = require 'ffi'
require 'pl'
require 'usefulFunctions'
require 'sys'
--require('mobdebug').start()

-- Title definition -----------------------------------------------------------
local title = [[
          _           _                         _   _ ______ _______
         (_)         | |                       | \ | |  ____|__   __|
          _ _ __   __| | ___   ___  _ __ ______|  \| | |__     | |
         | | '_ \ / _` |/ _ \ / _ \| '__|______| . ` |  __|    | |
         | | | | | (_| | (_) | (_) | |         | |\  | |____   | |
         |_|_| |_|\__,_|\___/ \___/|_|         |_| \_|______|  |_|

]]

-- Options --------------------------------------------------------------------
opt = lapp(title .. [[

Dataset's parameters
   --side    (default 46  ) Training and testing image's side length (max 256)
   --colour  (default true) True by default, allows to train on B&W if the flag is called
   --jitter  (default 0   ) Introduce random crop for loweing overfitting
   --distort                TODO
   --mmload                 Memory mapping when loading data. Use with small RAM
   --mm_threads (default 4) Number of threads use for mm
   --parts                  Use image parts instead of whole images
   --dropout (default 0)    Dropout in MLP. Set it to 0 for disabling it, 0.5 for "standard" working value
   --inputDO (default 0)    Input dropout. Set it to 0 for disabling it, 0.2 for "standard" working value

Learning parameters
   --learningRate      (default 5e-2)
   --learningRateDecay (default 1e-7)
   --weightDecay       (default 0   )
   --momentum          (default 0   )
   --batchSize         (default 128 )

Saving parameters and temporary data
Be carefull!!! you need to recompute temporary data when you change data options like width
To doing so, just call the flag <cleanRun>
   --cleanRun                                 Run without loading any previously stored data
   --save_dir        (default './results/'  )
   --temp_dir        (default './temp-data/')
   --subsample_name  (default 'indoor51'    ) Name of imagenet subsample. Possible options ('class51', 'elab')
   --data_sl         (default 'load'        ) Save data images once and then load prepared data from temp file ([load]|save)
   --mean_sl         (default 'load'        ) Save data mean once and then load prepared mean from temp file ([load]|save)

On screen output
   --plot                                  Plot training and testing accuracies
   --verbose                (default true) Show more output on screen
   --confusion_matrix_tab   (default 3   ) Number of tabs between numbers in print of confusion matrix
   --print_confusion_matrix                Print confusion matrix after training and testing
   --print_weight_stat      (default true) Print number of neuralnet weights which are lower 1e-5 and higher 1e+2

CUDA parameters
   --cuda  (default true)
   --devid (default 1   ) device ID (if using CUDA)

Other parameters
   --verify_statistics
   --save              (default true)
   --seed              (default 123 )
   --num_threads       (default 3   )
   --probe                            Prints to screen feature maps stats after every training iteration
   --debug                            Logs weights and gradients stats every epoch
   --network           (default N/A ) Load specified net from file. Default is "Not Available" -> new net is generated
]])

torch.setdefaulttensortype('torch.FloatTensor')

--allow write default false
for a,b in pairs(opt) do
   if (b == 'false') then opt[a] = false end
end

-- Aggiusting options ---------------------------------------------------------
opt.width  = opt.side
opt.height = opt.side
opt.side   = nil
if opt.cleanRun then
   opt.data_sl = 'save'
   opt.mean_sl = 'save'
end

-- Title ----------------------------------------------------------------------
if opt.verbose then print(title)

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

--set data paths
--data_folder = eex.datasetsPath() .. 'originalDataset/'
data_folder = '/media/DataDrive/Datasets/'
train_info_file = data_folder .. 'train-info-' .. opt.subsample_name .. '.t7' --file with labels, image sizes and paddings
test_info_file = data_folder .. 'test-info-' .. opt.subsample_name .. '.t7'

local testFolder = data_folder .. 'test-folder-' .. opt.subsample_name .. '/'
local trainFolder = data_folder .. 'train-folder-' .. opt.subsample_name .. '/'

if (paths.dirp(testFolder))  then
   test_data_file = testFolder .. 'test-data-' .. opt.subsample_name
else
   test_info_file = data_folder .. 'test-info-' .. opt.subsample_name .. '.t7'
end

if (paths.dirp(trainFolder))  then
   train_data_file = trainFolder .. 'train-data-' .. opt.subsample_name
else
   train_data_file = data_folder .. 'train-data-' .. opt.subsample_name .. '.t7'
end

--load classes
dofile('Data/indoor-classes.lua')
-------------------------------------------------------------------------------

torch.manualSeed(opt.seed)
torch.setnumthreads(opt.num_threads) --some of data scripts may change numthreads, so we need to do it here
print('Currently using ' .. torch.getnumthreads() .. ' threads')
os.execute('mkdir -p ' .. opt.save_dir) --create folder for saving results

--compute global mean and std
global_mean, global_std = get_global_mean_async(train_data_file, train_info_file, opt.save_dir, opt.mean_sl, opt.verbose)

--prepare data
if opt.mmload then
   --use memory mapping
   testData = load_data_mm(test_data_file, test_info_file)
   trainData = load_data_mm(train_data_file, train_info_file)
else
   --load all data at once
   opt.mm_threads = 1
   testData = load_data(test_data_file, test_info_file, 'test.t7', opt.data_sl)
   trainData = load_data(train_data_file, train_info_file, 'train.t7', opt.data_sl)
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
   local model, loss, dropout
   if opt.network == 'N/A' then
      model, logsoft, loss, dropout = get_model1() --(classifier.lua)
      local tmpFile = io.open('.pltStat', 'r')
      statFile:write(tmpFile:read('*all'))
      io.close(tmpFile)
      os.execute('rm .pltStat .pltStatData')
   else
      print('Loading network from file: ' .. opt.network)
      model, logsoft, loss, dropout = get_model2(opt.network)
   end
   collectgarbage() -- get rid of craps from the GPU's RAM

   --train classifier
   local train_acc, test_acc = train_and_test(trainData, testData, model, logsoft, loss, opt.plot, opt.verbose, dropout) --(train-and-test.lua)
   -------------------------------------------------------------------------------

   --return train and test accuracy
   return train_acc, test_acc

end

-- Logging statistics ----------------------------------------------------------
print '==> Set up Log'
if (opt.network == 'N/A') then
   -- Open file in re-write mode (NOT append)
   statFile = io.open(opt.save_dir .. 'stat.txt','w+')
   -- Writing title
   statFile:write(title)
else
   statFile = io.open(opt.save_dir .. 'stat.txt','a')

   statFile:write('\n')
   statFile:write('-------------------------------------------------------------------------------\n')
   statFile:write('------------------------------------ Restart ----------------------------------\n')
   statFile:write('-------------------------------------------------------------------------------\n')
end
-- Writing currect commit hash
statFile:write(string.format('Current commit hash: %s\n',sys.execute('git rev-parse HEAD')))
-- Collecting input arguments and writing them to file
local inputArg = ''
for i = 1,#arg do inputArg = inputArg .. ' ' .. arg[i]; end
statFile:write(string.format('User command line input:%s\n',inputArg))
statFile:flush()

-- Training and testing --------------------------------------------------------
print '==> Start Training'
train_acc, test_acc = run(trainData, testData)


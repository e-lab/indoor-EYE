-------------------------------------------------------------------------------
-- Main script for training neural network
-- Gregory Essertel, June 2014
-- based on the work of:
-- Artem Kuharenko
-- Alfredo Canziani, Feb 2014
-- Gregory Essertel, May 2014
-------------------------------------------------------------------------------

-- Require packages -----------------------------------------------------------
require 'torch'
require 'nnx'
require 'net-toolkit'
local lapp = require 'pl.lapp'
require 'sys'
require 'cutorch'
require 'cunn'
require 'dataset-manager'

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
--side            (default 128)        Training and testing image's side length (max 256)
--jitter          (default 0)          Introduce random crop for loweing overfitting
--dropout         (default 0)          Dropout in MLP. Set it to 0 for disabling it, 0.5 for "standard" working value
--inputDO         (default 0)          Input dropout. Set it to 0 for disabling it, 0.2 for "standard" working value
--subsample_name  (default 'indoor51') Name of imagenet subsample. Possible options ('class51', 'elab')

Learning parameters
--learningRate      (default 5e-2)
--learningRateDecay (default 1e-7)
--weightDecay       (default 0   )
--momentum          (default 0   )
--batchSize         (default 128 )

On screen output
--plot                                  Plot training and testing accuracies
--verbose                (default true) Show more output on screen

Output folder
--save_dir (default './results/') Folder where the stats and models are saved

CUDA parameters
--devid (default 1) device ID (if using CUDA)

Other parameters
--num_threads       (default 6   )
--probe                            Prints to screen feature maps stats after every training iteration
--network           (default N/A ) Load specified net from file. Default is "Not Available" -> new net is generated
]])

torch.setdefaulttensortype('torch.FloatTensor')

-- Aggiusting options ---------------------------------------------------------
-- Title ----------------------------------------------------------------------
if opt.verbose then
   print(title)

   -- Print options summary ------------------------------------------------------
   print('==> Options:')
   for a,b in pairs(opt) do print('     + ' .. a .. ':', b) end
   print()
end

-- Training with GPU ----------------------------------------------------------
if opt.verbose then
   print('==> switching to CUDA')
end
cutorch.setDevice(opt.devid)
cutorch.deviceReset()
if opt.verbose then
   print('==> using GPU #' .. cutorch.getDevice())
   print(cutorch.getDeviceProperties(opt.devid))
end

-- Loading functions ----------------------------------------------------------
if opt.verbose then
   print('==> Loading <training> and <testing> functions')
end
dofile('train-and-test-gpu.lua')

if opt.verbose then
   print('==> Loading <model> functions')
end
dofile('models.lua')
if opt.verbose then
   print('==> Loading <data> functions')
end

torch.manualSeed(123)
torch.setnumthreads(opt.num_threads) --some of data scripts may change numthreads, so we need to do it here

print('Currently using ' .. torch.getnumthreads() .. ' threads')
os.execute('mkdir -p ' .. opt.save_dir) --create folder for saving results

-- Datasets --------------------------------------------------------------------
local trainOpt = {}
trainOpt.side = opt.side
trainOpt.jitter = opt.jitter
trainOpt.batchSize = opt.batchSize
trainOpt.verbose = true
trainOpt.nbThreads = 4
trainOpt.normInput = true

local datasetExtractor = dmanager.TrainingExtractorAsync(opt.subsample_name, trainOpt)

--print train and test image sizes
if verbose then
   print(string.format('==> Train number of batches: %d, Batch size: %dx%dx%dx%d', datasetExtractor.getNbBatches(true), opt.colour, opt.side, opt.side))
   print(string.format('==> Test number of batches: %d, Batch size: %dx%dx%dx%d',  datasetExtractor.getNbBatches(false), opt.colour, opt.side, opt.side))
end

-- Loggers ----------------------------------------------------------------------
print '==> Set up Log'
local statFile
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
-- Writing current commit hash
statFile:write(string.format('Current commit hash: %s\n',sys.execute('git rev-parse HEAD')))
-- Collecting input arguments and writing them to file
local inputArg = ''
for i = 1,#arg do
   inputArg = inputArg .. ' ' .. arg[i];
end
statFile:write(string.format('User command line input:%s\n',inputArg))
statFile:flush()

-- Training and testing --------------------------------------------------------
print '==> Start Training'

--get classifier and loss function
local model, logsoft, loss, dropout
if opt.network == 'N/A' then
   model, logsoft, loss, dropout = get_model1(#datasetExtractor:getClasses(), statFile, true) --(classifier.lua)
   local tmpFile = io.open('.pltStat', 'r')
   statFile:write(tmpFile:read('*all'))
   io.close(tmpFile)
   os.execute('rm .pltStat .pltStatData')
else
   if (verbose) then
      print('Loading network from file: ' .. opt.network)
   end
   model, logsoft, loss, dropout = get_model2(opt.network, true)
end
collectgarbage() -- get rid of craps from the GPU's RAM

--train classifier
train_and_test(datasetExtractor, model, logsoft, loss, dropout, statFile) --(train-and-test.lua)

io.close(statFile)
-------------------------------------------------------------------------------

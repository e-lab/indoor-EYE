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
require 'nn'
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
--side     (default 128)        Training and testing image's side length (max 256)
--jitter   (default 0)          Introduce random crop for loweing overfitting
--dataset  (default 'indoor51') Name of imagenet subsample. Possible options ('class51', 'elab')

Model's parameters
--dropout         (default 0)          Dropout in MLP. Set it to 0 for disabling it, 0.5 for "standard" working value
--inputDO         (default 0)          Input dropout. Set it to 0 for disabling it, 0.2 for "standard" working value

Learning parameters
--learningRate      (default 5e-2)
--learningRateDecay (default 1e-7)
--weightDecay       (default 0   )
--momentum          (default 0   )
--batchSize         (default 128 )
--renorm            (default 0   )     If every weight of a kernel are equal, it is the maximum value. (0 means no renormalisation)

On screen output
--plot                                  Plot training and testing accuracies
--verbose                (default true) Show more output on screen

Output folder
--save_dir (string) Folder where the stats and models are saved

CUDA parameters
--devid (number) device ID (if using CUDA)

Other parameters
--num_threads       (default 6   )
--probe                            Prints to screen feature maps stats after every training iteration
--network           (default N/A ) Load specified net from file. Default is "Not Available" you have to feed a network!!! 
--classifier        (default N/A)  Load the classifier from file. Default is "Not Available" -> new net is generated
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

if (string.sub(opt.save_dir, -1) ~= '/') then
   opt.save_dir = opt.save_dir .. '/'
end
if (paths.dirp(opt.save_dir) and opt.classifier == 'N/A') then
   error(string.format("the folder %s already exists, to avoid conflicts delete it or change save_dir name", opt.save_dir))
end
os.execute('mkdir -p ' .. opt.save_dir) --create folder for saving results

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
   print('==> Initiazing loggers')
end
dofile('init-loggers.lua')

torch.manualSeed(123)
torch.setnumthreads(opt.num_threads) --some of data scripts may change numthreads, so we need to do it here

print('Currently using ' .. torch.getnumthreads() .. ' threads')

-- Datasets --------------------------------------------------------------------
dmanager.initPath(opt.devid)

local trainOpt = {}
trainOpt.side              = opt.side
trainOpt.trainJitter       = opt.jitter
trainOpt.trainJitterMode   = 'random'
trainOpt.batchSize         = opt.batchSize
trainOpt.nbThreads         = 4
trainOpt.mean              = 0
trainOpt.std               = 1
local datasetExtractor = dmanager.TrainingExtractorAsync(opt.dataset, trainOpt)

--print train and test image sizes
if verbose then
   print(string.format('==> Train number of batches: %d, Batch size: %dx%dx%dx%d', datasetExtractor.getNbBatches(true), 3, opt.side, opt.side))
   print(string.format('==> Test number of batches: %d, Batch size: %dx%dx%dx%d',  datasetExtractor.getNbBatches(false), 3, opt.side, opt.side))
end
-- Get loggers -----------------------------------------------------------------

local statFile, logger, ce_logger, logger_5 = getLoggers()

-- Writing title
statFile:write(title)

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
   assert(false)
else
   if (verbose) then
      print('Loading network from file: ' .. opt.network)
   end
   model = get_model2(opt.network, true)
   local no_layer = (model.modules[2]:size())
   sizeNeuron = model.modules[2].modules[no_layer].weight:size(2)
   model.modules[2].modules[no_layer] = nil
end

if opt.classifier == 'N/A' then 
   classifier, logsoft, loss = get_model3(sizeNeuron, #datasetExtractor:getClasses(), true) --(classifier.lua)
else 
   classifier, logsoft, loss = get_model2(opt.network, true)
end 

collectgarbage() -- get rid of craps from the GPU's RAM

--train
train_and_test(datasetExtractor, model, classifier, logsoft, loss, dropout, statFile, logger, ce_logger, logger_5) --(train-and-test.lua)
-------------------------------------------------------------------------------

-------------------------------------------------------------------------------
-- Main script for training neural network
-- Artem Kuharenko
-------------------------------------------------------------------------------

require 'torch'
require 'nn'
require 'torchffi'
require 'pl'
require 'usefulFunctions'

--lapp doesn't work with qlua...
opt = {}
opt.width = 46
opt.height = opt.width
opt.ncolors = 3

--learning
opt.learningRate = 5e-3
opt.learningRateDecay = 1e-7
opt.weightDecay = 0
opt.momentum = 0
opt.batchSize = 32
opt.classifier_size = 32
opt.niters = 20

--save/load data
opt.save_dir = './results/'
opt.temp_dir = './temp-data/'
--Be carefull!!! you need to save data when you change data options like width 
opt.data_sl = 'save' --save once and then load prepared data from temp file
opt.mean_sl = 'save' --save once and then load prepared mean from temp file

--output
opt.plot = false
opt.verbose = true
opt.confusion_matrix_tab = 3 --number of tabs between numbers in confusion matrix
opt.print_confusion_matrix = false

--other
opt.verify_statistics = false
opt.save = true

opt.cuda = true
opt.mmload = false --memory mapping when loading data. Use with small RAM
opt.seed = 123
opt.num_threads = 3
-------------------------------------------------------------------------------

if opt.cuda then
   require 'cutorch'
   require 'cunn'
end

dofile('train-and-test.lua')
dofile('models.lua')
dofile('Data/data-imagenet.lua') --imagenet data scripts
dofile('Data/data-process.lua') --data scripts

os.execute('mkdir -p ' .. opt.temp_dir) --create folder for temporary data 
-------------------------------------------------------------------------------

data_folder = eex.datasetsPath() .. 'imagenet2012/'
train_data_file = data_folder .. 'train-indoor-data-1300.t7'
train_info_file = data_folder .. 'train-indoor-info-1300.t7'
test_data_file = data_folder .. 'test-indoor-data-50.t7'
test_info_file = data_folder .. 'test-indoor-info-50.t7'

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


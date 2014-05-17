require 'pl'
require 'eex'
require 'torch'
ffi = require 'ffi'

opt = lapp([[
      --test                           exctract the test dataset with labels
      --train                          exctract the train dataset with labels
      --height         (default 128)
      --width          (default 128)
      --subsample_name (string)
]])


torch.setdefaulttensortype('torch.FloatTensor')
-- option which have to be set
opt.ncolors = 3

-- useless option
opt.batchSize = 1
opt.jitter = 0
global_mean = torch.Tensor(3):fill(0)
global_std = torch.Tensor(3):fill(1)


print(opt.height)

dofile('data-imagenet.lua')

data_folder = eex.datasetsPath() .. 'originalDataset/'
train_info_file = data_folder .. 'train-info-' .. opt.subsample_name .. '.t7' --file with labels, image sizes and paddings
test_info_file = data_folder .. 'test-info-' .. opt.subsample_name .. '.t7'

local testFolder = data_folder .. 'test-folder-' .. opt.subsample_name .. '/'
local trainFolder = data_folder .. 'train-folder-' .. opt.subsample_name .. '/'

if (paths.dirp(testFolder))  then
   test_data_file = testFolder .. 'test-data-' .. opt.subsample_name
else
   test_data_file = data_folder .. 'test-data-' .. opt.subsample_name .. '.t7'
   test_info_file = data_folder .. 'test-info-' .. opt.subsample_name .. '.t7'
end


if (paths.dirp(trainFolder))  then
   train_data_file = trainFolder .. 'train-data-' .. opt.subsample_name
else
   train_data_file = data_folder .. 'train-data-' .. opt.subsample_name .. '.t7'
end

if (opt.train) then
   local data = load_data(train_data_file, train_info_file, '', '')
   local d = {}
   d.data = data.data
   d.labels = data.labels
   print('Saving data in ' .. data_folder .. opt.subsample_name .. '-train.t7')
   torch.save(data_folder .. opt.subsample_name .. '-train.t7', d)
end

if (opt.test) then
   print('==> Test')
   local data = load_data(test_data_file, test_info_file, '', '')
   local d = {}
   d.data = data.data
   d.labels = data.labels
   print('Saving data in ' .. data_folder .. opt.subsample_name .. '-test.t7')
   torch.save(data_folder .. opt.subsample_name .. '-test.t7', d)
end

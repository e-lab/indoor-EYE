-------------------------------------------------------------------------------
-- Create subsamples of imagenet
-- Visualize train and test data
-- Save subsample classes in different folders. This helps to check data.
--
-- Artem Kuharenko, February 2014
-------------------------------------------------------------------------------
require 'pl'
require 'eex'

opt = lapp([[

   --show_test                           set true if you want to look at test images
   --show_train                          set true if you want to look at train images
   --subsample_test                      set true to create subsample of test imagenet images
   --subsample_train                     set true to create subsample of train imagenet images
   --save_test                           save test images. Helps to check data
   --save_train                          save train images. Helps to check data
   --subsample_name      (default kitchen ) name of imagenet subsample.
   --convert_class_names                    convert csv class file to torch format

   --sample_name         (default imagenet)

   --width               (default 46) width of data. Only for show and save functions.
   --height              (default 46) height of data. Only for show and save functions.
   --ncolors             (default 3 )
   --batchSize           (default 32)
   --jitter              (default 0 )

]])

--allow write default false
for a,b in pairs(opt) do
   if (b == 'false') then opt[a] = false end
end

--print options
print(opt)

dofile('data-imagenet.lua')
-----------------------------------------------------------------------------------------------
--set data paths
data_folder = eex.datasetsPath() .. 'originalDataset/'

--source data files
src_train_info = data_folder .. 'train-info-' .. opt.sample_name .. '.t7' --file with labels, image sizes and paddings
src_test_info = data_folder .. 'test-info-' .. opt.sample_name .. '.t7'

local testFolder = data_folder .. 'test-folder-' .. opt.sample_name .. '/'
local trainFolder = data_folder .. 'train-folder-' .. opt.sample_name .. '/'

if (paths.dirp(testFolder))  then
   src_test_data = testFolder .. 'test-data-' .. opt.sample_name
else
   src_test_data = data_folder .. 'test-data-' .. opt.sample_name .. '.t7'
end

if (paths.dirp(trainFolder))  then
   src_train_data = trainFolder .. 'train-data-' .. opt.sample_name
else
   src_train_data = data_folder .. 'train-data-' .. opt.sample_name .. '.t7'
end

--destination subsample filenames
folderTrain = data_folder .. 'train-folder-' .. opt.subsample_name
if (not paths.dirp(folderTrain)) then
   os.execute('mkdir ' .. folderTrain)
end

folderTest = data_folder .. 'test-folder-' .. opt.subsample_name
if (not paths.dirp(folderTest)) then
   os.execute('mkdir ' .. folderTest)
end
train_data_file = folderTrain .. '/train-data-' .. opt.subsample_name
train_info_file = data_folder .. 'train-info-' .. opt.subsample_name .. '.t7'
test_data_file = folderTest .. '/test-data-' .. opt.subsample_name
test_info_file = data_folder .. 'test-info-' .. opt.subsample_name .. '.t7'

if opt.convert_class_names then
   --convert class names from csv to torch table. Do this once.
   print(data_folder .. 'classes.csv', data_folder .. 'classes.th')
   csv2table(data_folder .. 'classes.csv', data_folder .. 'classes.th')
end

--load class names
imagenet_class_names = torch.load(data_folder .. 'classes.th')

--define classes and class_names
dofile('indoor-classes.lua')

if opt.subsample_test then
   --subsample test data of imagenet
   max_class_size = 50 --max number of images in each class
   filter_imagenet(src_test_data, src_test_info, test_data_file, test_info_file, classes, imagenet_class_names, max_class_size)
end

if opt.subsample_train then
   --subsample test data of imagenet
   max_class_size = 1300 --max number of images in each class
   filter_imagenet(src_train_data, src_train_info, train_data_file, train_info_file, classes, imagenet_class_names, max_class_size)
end

--global_mean and global_std are used in data loading functions
global_mean = {0, 0, 0}
global_std = {1, 1, 1}

if opt.save_test or opt.show_test then
   --load raw resized test photos
   testData = load_data(test_data_file, test_info_file)
end

if opt.save_train or opt.show_train then
   --load raw resized train photos
   trainData = load_data(train_data_file, train_info_file)
end

if opt.show_test then
   --show test classes
   show_classes(testData, 50, class_names)
end

if opt.show_train then
   --show test classes
   show_classes(trainData, 1300, class_names)
end

if opt.save_test then
   --save test photos of each class in separate folder
   verify_data(testData, classes, imagenet_class_names, 'test_photos/')
end

if opt.save_train then
   --save train photos of each class in separate folder
   verify_data(trainData, classes, imagenet_class_names, 'train_photos/')
end


-------------------------------------------------------------------------------
-- Create subsamples of imagenet
-- Visualize train and test data
-- Save subsample classes in different folders. This helps to check data.
--
-- Artem Kuharenko, February 2014
-------------------------------------------------------------------------------
require 'pl'
require 'torchffi'
require 'eex'

opt = lapp([[

   --show_test           (default false) set true if you want to look at test images
   --show_train          (default false) set true if you want to look at train images
   --subsample_test      (default false) set true to create subsample of test imagenet images
   --subsample_train     (default false) set true to create subsample of train imagenet images
   --save_test           (default false) save test images. Helps to check data
   --save_train          (default false) save train images. Helps to check data
   --subsample_name      (default elab ) name of imagenet subsample.
   --convert_class_names (default false) convert csv class file to torch format

   --src_test_data       (default test256m-data.t7 ) --file with compressed jpegs
   --src_test_info       (default test256m-info.t7 ) --file with labels, image sizes and paddings
   --src_train_data      (default train256m-data.t7)
   --src_train_info      (default train256m-info.t7)

   --width               (default 46) --width of data. Only for show and save functions. 
   --height              (default 46) --height of data. Only for show and save functions. 
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
path = eex.datasetsPath() .. 'imagenet2012/'

--source data files
src_train_data = path .. opt.src_train_data
src_train_info = path .. opt.src_train_info
src_test_data = path .. opt.src_test_data
src_test_info = path .. opt.src_test_info

--destination subsample filenames
train_data_file = path .. 'train-data-' .. opt.subsample_name .. '.t7'
train_info_file = path .. 'train-info-' .. opt.subsample_name .. '.t7'
test_data_file = path .. 'test-data-' .. opt.subsample_name .. '.t7'
test_info_file = path .. 'test-info-' .. opt.subsample_name .. '.t7'

if opt.convert_class_names then
   --convert class names from csv to torch table. Do this once.
   print(path .. 'classes.csv', path .. 'classes.th')
   csv2table(path .. 'classes.csv', path .. 'classes.th')
end

--load class names
imagenet_class_names = torch.load(path .. 'classes.th')

--load classes
dofile('indoor-classes.lua')
class_names = {}
for i = 1, #classes do class_names[i] = classes[i][1] end

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


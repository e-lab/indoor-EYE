-------------------------------------------------------------------------------
-- Examples of using data functions
-- Artem Kuharenko
-------------------------------------------------------------------------------
require 'pl'
require 'torchffi'
opt = lapp([[

   --show_test           (default false ) 
   --show_train          (default false ) 
   --load_test           (default false ) 
   --load_train          (default false ) 
   --async_load_example  (default false ) 
   --subsample_test      (default false ) 
   --subsample_train     (default false )
   --subsample_classes   (default 'elab') 
   --save_test           (default false ) 
   --save_train          (default false ) 

   --src_test_data       (default test256m-data.t7 ) 
   --src_test_info       (default test256m-info.t7 ) 
   --src_train_data      (default train256m-data.t7 ) 
   --src_train_info      (default train256m-info.t7 ) 
   --test_data_file      (default test-data-elab.t7 ) 
   --test_info_file      (default test-info-elab.t7 ) 
   --train_data_file     (default train-data-elab.t7 ) 
   --train_info_file     (default train-info-elab.t7 ) 

   --width               (default 46    ) 
   --height              (default 46    ) 
   --ncolors             (default 3     ) 
   --batchSize           (default 32    ) 
   --jitter               (default 0    ) 

]])

--allow write default false
for a,b in pairs(opt) do
   if (b == 'false') then opt[a] = false end
end

--print options
print(opt)

dofile('data-imagenet.lua')
data_folder = '/home/artem/datasets/imagenet2012/'
-----------------------------------------------------------------------------------------------
if opt.async_load_example then
   --async load batch example

   testData = load_imagenet_async(data_folder .. 'test256m-data.t7', data_folder .. 'test256m-info.t7')
   testData.prepareBatch(1) --this is done async in another thread
   b = testData.copyBatch() --now you get first batch

end
------------------------------------------------------------------------------------------------

src_train_data = data_folder .. opt.src_train_data
src_train_info = data_folder .. opt.src_train_info
src_test_data = data_folder .. opt.src_test_data
src_test_info = data_folder .. opt.src_test_info

train_data_file = data_folder .. opt.train_data_file
train_info_file = data_folder .. opt.train_info_file
test_data_file = data_folder .. opt.test_data_file
test_info_file = data_folder .. opt.test_info_file

if opt.convert_class_names then
   --convert class names from csv to torch table. Do this once and the comment
   print(data_folder .. 'classes.csv', data_folder .. 'classes.th')
   csv2table(data_folder .. 'classes.csv', data_folder .. 'classes.th')
end

--load class names
imagenet_class_names = torch.load(data_folder .. 'classes.th')

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

global_mean = {0, 0, 0}
global_std = {1, 1, 1}

if opt.load_test or opt.show_test then
   --load raw resized test photos
   testData = prepare_sync(test_data_file, test_info_file)
end

if opt.load_train or opt.show_train then
   --load raw resized train photos
   trainData = prepare_sync(train_data_file, train_info_file)
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


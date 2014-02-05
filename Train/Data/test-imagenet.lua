require 'torchffi'
dofile('data-imagenet.lua')

data_folder = '/home/artem/datasets/imagenet2012/'
-----------------------------------------------------------------------------------------------
if 1 == 0 then
   --async load batch example

   testData = load_imagenet_async(data_folder .. 'test256m-data.t7', data_folder .. 'test256m-info.t7')
   testData.prepareBatch(1) --this is done async in another thread
   b = testData.copyBatch() --now you get first batch

end
------------------------------------------------------------------------------------------------

src_data = data_folder .. 'test256m-data.t7'
src_info = data_folder .. 'test256m-info.t7'
dst_data = data_folder .. 'test-indoor-data-50.t7'
dst_info = data_folder .. 'test-indoor-info-50.t7'

--convert class names from csv to torch table. Do this once and the comment
--print(data_folder .. 'classes.csv', data_folder .. 'classes.th')
--csv2table(data_folder .. 'classes.csv', data_folder .. 'classes.th')

--load class names
imagenet_class_names = torch.load(data_folder .. 'classes.th')

dofile('indoor-classes.lua')
class_names = {}
for i = 1, #indoor_classes do class_names[i] = indoor_classes[i][1] end

--filtering example
max_class_size = 50 --max number of images in each class
--filter_imagenet2(src_data, src_info, dst_data, dst_info, indoor_classes, imagenet_class_names, max_class_size)

--load raw resized photos
opt = {}
opt.width = 46
opt.height = 46
opt.batchSize = 32
opt.ncolors = 3
--testData = load_raw_imagenet(dst_data, dst_info)
global_mean = {0, 0, 0}
global_std = {1, 1, 1}
testData = prepare_sync(dst_data, dst_info)

--show objects of different classes
--show_classes(testData, 50, class_names)

--save photos of each class in separate folder 
verify_data(testData, indoor_classes, imagenet_class_names, 'photos/')


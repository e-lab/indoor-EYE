dofile('data-imagenet.lua')

data_folder = '/home/artem/neuflow/datasets/imagenet2012/'
-----------------------------------------------------------------------------------------------
if 1 == 0 then
	--async load batch example

	testData = load_imagenet_async(data_folder .. 'test256m-data.t7', data_folder .. 'test256m-info.t7')
	testData.prepareBatch(1) --this is done async in another thread
	b = testData.copyBatch() --now you get first batch 

end
------------------------------------------------------------------------------------------------

classes = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
class_names = {'1','2','3','4','5','6','7','8','9','10'}
src_data = data_folder .. 'test256m-data.t7'
src_info = data_folder .. 'test256m-info.t7'
dst_data = data_folder .. 'test-10cl-data.t7'
dst_info = 	data_folder .. 'test-10cl-info.t7'
--filtering example 
filter_imagenet(src_data, src_info, dst_data, dst_info, classes, class_names)

--load raw resized photos
opt = {}
opt.width = 46
opt.height = 46
testData = load_raw_imagenet(dst_data, dst_info)

--show photos
show_classes(testData)

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

src_data = data_folder .. 'test256m-data.t7'
src_info = data_folder .. 'test256m-info.t7'
dst_data = data_folder .. 'test-10cl-data.t7'
dst_info = 	data_folder .. 'test-10cl-info.t7'

--convert class names from csv to torch table. Do this once and the comment
csv2table(data_folder .. 'classes.csv', data_folder .. 'classes.th')

--load class names
class_names = torch.load(data_folder .. 'classes.th')

--filtering example 
classes = {991, 992, 993, 994, 995, 996, 997, 998, 999, 1000} --class ids which we want to select
filter_imagenet(src_data, src_info, dst_data, dst_info, classes, class_names)

--load raw resized photos
opt = {}
opt.width = 46
opt.height = 46
testData = load_raw_imagenet(dst_data, dst_info)

--show objects of different classes
show_classes(testData, 100)

----------------------------------------------------------------------
-- Load imagenet dataset
-- Artem Kuharenko
-- Alfredo Canziani, Feb 2014
-- Use Clement Farabet scripts
----------------------------------------------------------------------

require 'torch'
require 'llthreads'
require 'sys'
require 'image'

function extract_square_patch(sample)

   -- extract square patch
   local size = math.min(sample:size(2), sample:size(3))
   local t = math.floor((sample:size(2) - size)/2 + 1)
   local l = math.floor((sample:size(3) - size)/2 + 1)
   local b = t + size - 1
   local r = l + size - 1
   sample = sample[{ {},{t,b},{l,r} }]

   return sample

end

----------------------------------------------------------------------
--scripts for async load with memory mapping

-- dimensions
--local c,h,w = 3, math.abs(opt.receptive), math.abs(opt.receptive)
local c,h,w=3, opt.height, opt.width
local bs = opt.batchSize

dofile('Data/load_data_multi.lua')

function load_data(data_file, info_file, sfile, fact)
   -- 1. Load and decompress imagenet data
   --With opt.parts option store original images (3, 256, 256), else store resized images

   -- 2. Define prepareBatch and copyBatch functions
   --prepareBatch: creates next batch for training or testing
   --copyBatch:    returns prepared batch
   -------------------------------------------------------------------------------

   local samples = torch.FloatTensor(opt.batchSize, opt.ncolors, opt.height, opt.width) --batch images
   local targets = torch.FloatTensor(opt.batchSize)                                     --batch labels

   local samplesCUDA = {}  --cuda batch images
   local targetsCUDA = {}  --cuda batch labels
   if opt.cuda then
      samplesCUDA = samples:clone():cuda()
      targetsCUDA = targets:clone():cuda()
   end

   print('==> Getting data')

   --here loaded data is stored
   local data = {}

   if fact == 'load' and paths.filep(opt.temp_dir .. sfile) then
      --load previously saved data

      local f = opt.temp_dir .. sfile
      print('======> Loading previously saved data from file ' .. f)
      data = torch.load(f)

   else

      -------------------------------------------------------------------------------
      --define sizes

      local w = opt.width  --final data width
      local h = opt.height --final data height

      local sw = opt.width --stored data width
      local sh = opt.height --stored data height

      if opt.parts then
         --store big images
         sw = 128
         sh = 128
      end

      if opt.jitter > 0 and not opt.parts then
         --store larger images in case of jitter
         sw = sw + opt.jitter
         sh = sh + opt.jitter
      end
      -------------------------------------------------------------------------------
      --load and decompress images

      print('=====> Loading info data from file: ' .. info_file)
      local dataset = torch.load(info_file) --data from info_file: labels, image sizes and offsets
      local jpegs = {}
      local jpegs_p = {}

      if (dataset.file_range) then
         number_file = dataset.file_range:size(1)
         for file = 1, number_file do
            jpegs[file] = torch.ByteStorage(data_file .. '-file' .. file .. '.t7')
            jpegs_p[file] = ffi.cast('unsigned char *', ffi.cast('intptr_t', torch.data(jpegs[file])))
         end
      else
         jpegs[1] = torch.ByteStorage(data_file) --compressed images
         jpegs_p[1] = ffi.cast('unsigned char *', ffi.cast('intptr_t', torch.data(jpegs[1])))
         dataset.data_p[1] = tonumber(ffi.cast('intptr_t', torch.data(dataset.data[1])))
      end

      local offsets_p = ffi.cast('unsigned long *', ffi.cast('intptr_t', torch.data(dataset.offsets)))
      local sizes_p   = ffi.cast('unsigned long *', ffi.cast('intptr_t', torch.data(dataset.sizes)))
      local file_number_p = ffi.cast('unsigned int *', ffi.cast('intptr_t', torch.data(dataset.file_number)))
      local gm = require 'graphicsmagick'

      local n = dataset.labels:size(1)
      data.data = torch.FloatTensor(n, opt.ncolors, sh, sw) --stored images
      data.labels = torch.Tensor(n)
      data.imagenet_labels = torch.Tensor(n)

      print('=====> Loading and decompressing jpegs from file: ' .. data_file)
      for i = 1, n do

         if (i == 1 or i == n or i%50 == 0) then
            xlua.progress(i, n)
         end

         local offset = tonumber(offsets_p[i-1] - 1) --offset of compressed image
         local size = tonumber(sizes_p[i-1])         --size of compressed image in bytes
         local file = tonumber(file_number_p[i-1])
         local jpegblob = jpegs_p[file] + offset           --pointer to compressed image

         --decompress image
         local im = gm.Image():fromBlob(jpegblob,size):toTensor('float','RGB','DHW',true)

         --extract square patch (256x256)
         im = extract_square_patch(im)

         --scale image
         --if not opt.parts then
         im = image.scale(im, sw, sh)
         --end

         --global normalization
         for j = 1, opt.ncolors do
            im[j]:add(-global_mean[j])
            im[j]:div(global_std[j])
         end

         data.data[i] = im

      end

      data.classes = dataset.classes
      data.labels = dataset.labels --subsample label id
      data.imagenet_labels = dataset.imagenet_labels --original imagenet label

      if fact == 'save' then
         --save data

         local f = opt.temp_dir .. sfile
         print('======> saving data to file ' .. f)
         torch.save(f, data)

      end

   end
   -------------------------------------------------------------------------------

   local n = data.labels:size(1)
   dataset.shuffle = torch.randperm(n):type('torch.LongTensor')
   data.keep = {
      shuffle = shuffle,
      samples = samples,
      targets = targets
   }

   local function prepareBatch(idx, istest)
      -- prepare next batch

      local bs = opt.batchSize

      for i = 1, bs do

         local j = dataset.shuffle[(idx - 1) * bs + i]
         local x1 = 1
         local y1 = 1

         if opt.jitter > 0 then
            --select random shifted subimage
            x1 = math.floor(torch.uniform(opt.jitter + 1))
            y1 = math.floor(torch.uniform(opt.jitter + 1))

         end

         if opt.parts then

            local x1 = math.floor(torch.uniform(1, data.data:size(4) - w + 1))
            local y1 = math.floor(torch.uniform(1, data.data:size(3) - h + 1))

         end

         samples[i] = data.data[j][{{},{y1, y1 + h - 1}, {x1, x1 + w - 1}}]
         targets[i] = data.labels[j]

      end

   end

   local function copyBatch(t, ims, tgs)
      ims:copy(samples)
      tgs:copy(targets)
   end

   dataset.newShuffle = function()
      dataset.shuffle:copy(torch.randperm(nSamples):type('torch.LongTensor'))
   end

   dataset.nBatches = math.floor(data.data:size(1) / opt.batchSize)

   -- augment dataset:
   data.copyBatch = copyBatch
   data.prepareBatch = prepareBatch

   return data

end

----------------------------------------------------------------------
--script for filtering imagenet
function create_imagenet_map(new_class, n)
   --maps imagenet classes to new class table
   --n - number of classes in imagenet

   --map
   local m = torch.Tensor(n, 2):zero()

   for i = 1, #new_class do
      local subclasses = new_class[i][2]
      for j = 1, #subclasses do
         local c = subclasses[j]
         --print(c)
         m[c][1] = i
         m[c][2] = j
      end
   end

   return m

end

function filter_imagenet(src_data, src_info, dst_data, dst_info, new_classes, imagenet_class_names, max_class_size)

   --load src data
   print('Loading src data')
   local d = torch.load(src_info)

   --map imagent ids to new class table
   local map_imagenet_id = create_imagenet_map(new_classes, 1000)

   --calc size of new data
   print('Calculating size of new data')
   local new_data_size = {0} --size of each new file in bytes
   local new_data_n = {0} --number of images in each new file
   local total_num = 0
   local idxs = {} --imagenet indexes of new images
   local class_size = torch.Tensor(#new_classes):zero() --current number of photos in each class
   local numFile = 1
   local MAX_SIZE = 512*1024*1024
   local MAX_NUM = 25000

   torch.manualSeed(1)
   local shuffle = torch.randperm(d.labels:size(1))

   for i = 1, d.labels:size(1) do

      if (i == 1 or i == d.labels:size(1) or i%10 == 0) then
         xlua.progress(i, d.labels:size(1))
      end

      local si = shuffle[i]
      local label = d.labels[si]
      local i1 = map_imagenet_id[label][1]

      if (new_data_n[numFile] == MAX_NUM or new_data_size[numFile] + d.sizes[si] > MAX_SIZE) then
         numFile = numFile + 1
         new_data_n[numFile] = 0
         new_data_size[numFile] = 0
      end

      if i1 > 0 and class_size[i1] < max_class_size then
         --image in new data

         new_data_n[numFile] = new_data_n[numFile] + 1
         total_num = total_num + 1
         new_data_size[numFile] = new_data_size[numFile] + d.sizes[si]
         table.insert(idxs, si)
         class_size[i1] = class_size[i1] + 1

      end

   end
   print('Number of samples:')
   print(new_data_n)
   print('Data size:')
   print(new_data_size)

   --allocate memory for new data
   print('Allocating memory')
   local labels = torch.LongTensor(total_num) --new labels
   local imagenet_labels = torch.LongTensor(total_num) --imagenet labels
   local sizes = torch.LongTensor(total_num) --new sizes
   local offsets = torch.LongTensor(total_num) --new offsets
   local file_number = torch.IntTensor(total_num) -- new file number
   local file_range = torch.IntTensor(#new_data_n, 2) -- new range

   local jpegs = {}
   local t_jpegs = {}
   for file = 1, d.file_range:size(1) do
      jpegs[file] = torch.ByteStorage(src_data .. '-file' .. file .. '.t7')
      t_jpegs[file] = torch.ByteTensor(jpegs[file])
   end

   --copy data
   print('Processing ' .. #new_data_n .. ' files')
   local offset = 1
   local global_index = 1

   for file = 1, #new_data_n do
      print('Copying file #' .. file)

      local t_new_jpegs = torch.ByteTensor(new_data_size[file])
      local offset = 1

      -- update starting point of the file
      file_range[file][1] = global_index

      for i = 1, new_data_n[file] do
         if (i == 1 or i == new_data_n[file] or i%10 == 0) then
         xlua.progress(i, new_data_n[file])
      end

         local j = idxs[global_index]
         local imagenet_label = d.labels[j]

         -- update info
         labels[global_index] = map_imagenet_id[imagenet_label][1]
         imagenet_labels[global_index] = imagenet_label
         sizes[global_index] = d.sizes[j]
         offsets[global_index] = offset
         file_number[global_index] = file

         -- update data
         t_new_jpegs[{{offset, offset + d.sizes[j] - 1}}] = t_jpegs[d.file_number[j]][{{d.offsets[j], d.offsets[j] + d.sizes[j] - 1}}]

         -- update auxilaries variables
         offset = offset + sizes[global_index]
         global_index = global_index + 1
      end

      file_range[file][2] = global_index - 1

      print('Saving file #' .. file)
      torch.save(dst_data .. '-file'.. file .. '.t7', torch.ByteTensor(new_data_size[file] - 107))
      local mmjpegs = torch.ByteStorage(dst_data .. '-file' .. file .. '.t7', true)
      mmjpegs:copy(t_new_jpegs:storage())

      -- nil temporary data
      t_new_jpegs = nil
      mmjpegs = nil

      collectgarbage()
   end

   --save data
   print 'Save info'
   local info = {}
   info.offsets = offsets
   info.sizes = sizes
   info.labels = labels
   info.imagenet_labels = imagenet_labels
   info.file_range = file_range
   info.file_number = file_number

   torch.save(dst_info, info)

end

function save_images(ims, fname, w)
   --save several images in one file.

   local n = ims:size(1)
   local w = w or 600
   local pad = 4
   local nrow = math.floor(w / (opt.width + pad))
   local ncol = math.floor(n / nrow) + 2
   local h = (opt.height + pad) * ncol
   local im = torch.Tensor(3, h, w):zero()

   local x = 1
   local y = 1

   for k = 1, n do

      if x + opt.width + pad >= w then
         x = 1
         y = y + opt.height + pad
      end

      im[{{}, {y, y + opt.height - 1}, {x, x + opt.width - 1}}] = ims[k]
      x = x + opt.width + pad

   end

   image.save(fname, im)

end

function verify_data(data, classes, imagenet_class_names, folder)
   --saves data in separate folders. Each class in separate folder.

   --map imagent ids to new class table
   map_imagenet_id = create_imagenet_map(classes, 1000)

   --create new classes structure
   bd = {}
   for i = 1, #classes do

      --print(i)
      local subclasses = classes[i][2]
      bd[i] = {}
      bd[i].name = classes[i][1]
      bd[i].subclasses = {}

      for j = 1, #subclasses do

         local sc = {}
         sc.imagenet_id = subclasses[j]
         sc.imagenet_name = imagenet_class_names[sc.imagenet_id]
         sc.size = 0
         sc.current_image = 0
         bd[i].subclasses[j] = sc

      end

   end

   --calc number of images in each class
   print('Calc number of images in each class')
   for i = 1, data.labels:size(1) do

      xlua.progress(i, data.labels:size(1))
      local id = data.imagenet_labels[i]
      local i1 = map_imagenet_id[id][1]
      local i2 = map_imagenet_id[id][2]
      if i1 > 0 then
         bd[i1].subclasses[i2].size = bd[i1].subclasses[i2].size + 1
      end

   end

   --create tensors for images
   for i = 1, #classes do
      local subclasses = bd[i].subclasses
      for j = 1, #subclasses do

         bd[i].subclasses[j].images = torch.Tensor(bd[i].subclasses[j].size, 3, opt.height, opt.width):zero()
      end
   end

   --copy images
   print('Copy images')
   for i = 1, data.labels:size(1) do

      xlua.progress(i, data.labels:size(1))
      local id = data.imagenet_labels[i]
      local i1 = map_imagenet_id[id][1]
      local i2 = map_imagenet_id[id][2]
      if i1 > 0 then
         local j = bd[i1].subclasses[i2].current_image + 1
         bd[i1].subclasses[i2].images[j] = data.data[i]
         bd[i1].subclasses[i2].current_image  = j
      end

   end

   --save to folder
   print('Saving images')
   os.execute('mkdir -p ' .. folder)
   for i = 1, #bd do

      xlua.progress(i, #bd)
      local class_folder = folder .. i .. '_' .. bd[i].name
      os.execute('mkdir -p ' .. class_folder)
      local subclasses = bd[i].subclasses

      for j = 1, #(bd[i].subclasses) do

         --  print(i .. ' ' .. j)
         if bd[i].subclasses[j].size > 0 then

            local fname = class_folder .. '/' .. j .. '_' .. subclasses[j].imagenet_name .. '_' .. subclasses[j].imagenet_id ..'.jpg'
            save_images(bd[i].subclasses[j].images, fname)

         end

      end

   end

end

function show_classes(data, k, class_names)

   local k = k or 100
   local ivch = data.data:size(2)
   local ivhe = data.data:size(3)
   local ivwi = data.data:size(4)

   local disp_ims = {}
   local nclasses = #class_names

   local nims = torch.Tensor(nclasses)

   for i = 1, nclasses do
      disp_ims[i] = torch.Tensor(k, ivch, ivhe, ivwi)
      nims[i] = 0
   end

   local ok = 0
   local i = 0

   while (ok < nclasses) and (i < data.data:size(1)) do

      i = i + 1
      local cl = data.labels[i]

      if (nims[cl] < k) then

         nims[cl] = nims[cl] + 1
         disp_ims[cl][nims[cl]] = data.data[i]
         if nims[cl] == k then
            ok = ok + 1
         end

      end

   end

   for i = 1, nclasses do
      disp_ims[i] = disp_ims[i][{{1, nims[i]}}]
   end

   for i = 1, nclasses do
      image.display({ image = disp_ims[i], legend = class_names[i], padding=4 })
   end

end
-----------------------------------------------------------------------------

function csv2table(csv_file, out_file)
   --convert class names from csv file to lua table

   local csv = require 'csv'
   torch.setdefaulttensortype('torch.FloatTensor')

   local class_names = {}

   -- load csv
   local f = csv.open(csv_file)

   for fields in f:lines() do

      i = tonumber(fields[1])
      if i then
         label = fields[3]
         class_names[i] = label
      else
         print 'skip'
      end

   end

   -- save tables
   torch.save(out_file, class_names)
   print('==> Saved ' .. out_file)

end

----------------------------------------------------------------------
-- Load image net data.
--
-- Clement Farabet, Artem Kuharenko
----------------------------------------------------------------------

require 'torch'
require 'llthreads'
require 'torchffi'
require 'sys'
require 'image'

opt = opt or {}
opt.batchSize = opt.batchSize or 128
opt.receptive = 256
opt.distort = false
opt.jitter = 0
opt.gm = 0
opt.lm = false
opt.ls = false
----------------------------------------------------------------------
--scripts for async load with memory mapping

-- dimensions
--local c,h,w = 3, math.abs(opt.receptive), math.abs(opt.receptive)
local c,h,w=3, opt.height, opt.width
local bs = opt.batchSize

function prepare(dataset)
   -- locals:
   local nsamples = dataset.labels:size(1)
   local samples = torch.FloatTensor(bs, c, h, w)
   local targets = torch.FloatTensor(bs)
   local jpegs = dataset.data
   local offsets = dataset.offsets
   local sizes = dataset.sizes
   local labels = dataset.labels
	--local shuffle = torch.randperm(nsamples):type('torch.LongTensor')

	shuffle = torch.range(1, nsamples):type('torch.LongTensor')

   -- com is a shared state between the main thread and
   -- the batch thread, it holds two variables:
   -- com[0]: the index of the batch to generate (if 0, then the thread is idle)
   -- com[1]: whether the batch is used for test or not (1 for test)
   local com = ffi.new('unsigned long[3]', {0,0})
   
   -- type:
--   local samplesCUDA = samples:clone():cuda()
--   local targetsCUDA = targets:clone():cuda()

   -- size:
   local function size()
      return math.floor(nsamples/bs)
   end

   -- block:
   local block = function(...)
      -- libs
      require 'sys'
      require 'torch'
      require 'torchffi'
      local gm = require 'graphicsmagick'
		require 'image'

      --local distort = require 'distort'

      -- args:
      local args = {...}
      local nsamples = args[2]
      local bs = args[3]
      local c = args[4]
      local h = args[5]
      local w = args[6]
      local distorton = args[7]
      local jitter = args[8]
      local receptive = args[9]
      local global_mean = args[10]
      local local_mean = args[11]
      local local_std = args[12]
      local shuffle_p = ffi.cast('unsigned long *', args[13])
      local samples_p = ffi.cast('float *', args[14])
      local targets_p = ffi.cast('float *', args[15])
      local jpegs_p   = ffi.cast('unsigned char *', args[16])
      local offsets_p = ffi.cast('unsigned long *', args[17])
      local sizes_p   = ffi.cast('unsigned long *', args[18])
      local labels_p  = ffi.cast('unsigned long *', args[19])
      local com       = ffi.cast('unsigned long *', args[20])


      -- map samples batch to given pointer:
      local samplesStorage = torch.FloatStorage(bs*c*h*w, tonumber(ffi.cast('intptr_t',samples_p)))
      local samples = torch.FloatTensor(samplesStorage):resize(bs,c,h,w)

      -- process batches in a loop:
      while true do
         -- next batch?
         while tonumber(com[0]) == 0 do
            sys.sleep(.005)
         end
         local idx = tonumber(com[0])

         -- train or test?
         local test = (com[1] == 1)

         -- zero batch
         samples:zero()

         -- process batch:
         for i = 0,bs-1 do
            -- offsets:
            local start = ((idx-1)*bs + i) % nsamples
            local ii = shuffle_p[start] - 1
            
            -- decode jpeg:
            local offset = tonumber(offsets_p[ii] - 1)
            local size = tonumber(sizes_p[ii])
            local jpegblob = jpegs_p + offset
            local sample = gm.Image():fromBlob(jpegblob,size):toTensor('float','RGB','DHW',true)
				
	         -- distort sample
            if distorton and not test then
               -- rot + flip + scale:
               sample = distort(sample, 0.25,.33,.5)
            end

            -- extract square patch
            local size = math.min(sample:size(2), sample:size(3))
            local t = math.floor((sample:size(2) - size)/2 + 1)
            local l = math.floor((sample:size(3) - size)/2 + 1)
            local b = t + size - 1
            local r = l + size - 1
            sample = sample[{ {},{t,b},{l,r} }]

				sample = image.scale(sample, w, h)

            -- extract sub-patch, with optional jitter:
            local size = math.min(sample:size(2), sample:size(3))
            local t = math.floor((sample:size(2) - h)/2 + 1)
            local l = math.floor((sample:size(3) - w)/2 + 1)
            if jitter > 0 and not test then
               t = t + math.floor(torch.uniform(-jitter/2,jitter/2))
               l = l + math.floor(torch.uniform(-jitter/2,jitter/2))
            end
            local b = t + h - 1
            local r = l + w - 1
            sample = sample[{ {},{t,b},{l,r} }]
            
            -- save sample:
            samples[i+1] = sample
            
            -- normalize sample
            samples[i+1]:add(-global_mean) 
            if local_mean then
               local mean = samples[i+1]:mean()
               samples[i+1]:add(-mean) 
            end
            if local_std then
               local std = samples[i+1]:std()
               samples[i+1]:div(std)
            end
            
            -- label:
            targets_p[i] = labels_p[ii]
         end

         -- reset com (tells the main thread that the batch is ready)
         com[0] = 0
      end
   end

   -- dispatch:
   local thread = llthreads.new(string.dump(block),
      nil,
      nsamples, bs, c, h, w,
      opt.distort,
      opt.jitter,
      opt.receptive,
      opt.gm,
      opt.lm,
      opt.ls,
      tonumber(ffi.cast('intptr_t', torch.data(shuffle))),
      tonumber(ffi.cast('intptr_t', torch.data(samples))),
      tonumber(ffi.cast('intptr_t', torch.data(targets))),
      tonumber(ffi.cast('intptr_t', torch.data(jpegs))),
      tonumber(ffi.cast('intptr_t', torch.data(offsets))),
      tonumber(ffi.cast('intptr_t', torch.data(sizes))),
      tonumber(ffi.cast('intptr_t', torch.data(labels))),
      tonumber(ffi.cast('intptr_t', com))
   )
   thread:start(true)
   
   -- keep references alive (the GC collects them otherwise,
   -- because it doesn't know that the thread needs them...)
   dataset.keep = {
      shuffle = shuffle,
      samples = samples,
      targets = targets,
      com = com,
      jpegs = jpegs,
      offsets = offsets,
      sizes = sizes,
      labels = labels,
   }

   -- prepare batch, asynchronously
   local function prepareBatch(idx, istest)
      -- request new batch!
      com[1] = (istest and 1) or 0
      com[0] = idx
   end

   -- copy batch to GPU
   local function copyBatch()
      -- wait on batch ready
      while tonumber(com[0]) > 0 do
         sys.sleep(.005)
      end

      -- move to CUDA
      --samplesCUDA:copy(samples)
     -- targetsCUDA:copy(targets)
      
      -- done
      return samples:clone(), targets:clone()
   end

   -- augment dataset:
   dataset.copyBatch = copyBatch
   dataset.prepareBatch = prepareBatch
   dataset.size = size
end

function load_imagenet_async(data_file, info_file)

	local imData = torch.load(info_file)
	imData.data = torch.ByteStorage(data_file)
	imData.n = imData.labels:size(1)
	imData.nbatches = math.floor(imData.n / opt.batchSize)	
	prepare(imData)


	return imData

end

----------------------------------------------------------------------
--script for filtering imagenet
function filter_imagenet(src_data_file, src_info_file, dst_data_file, dst_info_file, classes, class_names)

	--specify classes which we want to select

	--------------------------------------------------------------------------------------

	local d = torch.load(src_info_file)
	local jpegs = torch.ByteStorage(src_data_file)

	local classes_set = {}
	for i = 1, #classes do
		classes_set[classes[i]] = true
	end


	print('calculating size of selected data')
	local new_data_size = 0
	local new_data_n = 0
	local idxs = {}

	for i = 1, d.labels:size(1) do

		xlua.progress(i, d.labels:size(1))

		local label = d.labels[i]

		if classes_set[label] then

			new_data_n = new_data_n + 1
			new_data_size = new_data_size + d.sizes[i]
			table.insert(idxs, i)

		end

	end

	print('number of samples: ' .. new_data_n .. ', data size: ' .. new_data_size)

	print('allocating memory')
	local new_data = {}
	new_data.labels = torch.LongTensor(new_data_n)
	new_data.sizes = torch.LongTensor(new_data_n)
	new_data.offsets = torch.LongTensor(new_data_n)
	new_data.classes = class_names
	local t_new_jpegs = torch.ByteTensor(new_data_size)

	print('copy data')
	local offset = 1
	local t_jpegs = torch.ByteTensor(jpegs)

	for i = 1, new_data_n do

		xlua.progress(i, new_data_n)

		local j = idxs[i]
		new_data.labels[i] = d.labels[j]
		new_data.sizes[i] = d.sizes[j]
		new_data.offsets[i] = offset
		offset_old = offset 	
		offset = offset + new_data.sizes[i]	
	
		t_new_jpegs[{{offset_old, offset - 1}}] = t_jpegs[{{d.offsets[j], d.offsets[j] + d.sizes[j] - 1}}]

	end

	print('saving data')
	torch.save(dst_data_file, torch.ByteTensor(new_data_size - 107))
	local mmjpegs = torch.ByteStorage(dst_data_file, true)
	mmjpegs:copy(t_new_jpegs:storage())

	torch.save(dst_info_file, new_data)	

end
----------------------------------------------------------------------

--scripts for loading raw imagenet images from memory-mapped file
function load_raw_imagenet(src_data_file, src_info_file, class_names)

	print('loading raw imagenet')

	local opt = opt or {}
	opt.width = opt.width or 46
	opt.height = opt.height or 46

	local d = torch.load(src_info_file)
	local jpegs = torch.ByteStorage(src_data_file)

	local jpegs_p   = ffi.cast('unsigned char *', ffi.cast('intptr_t', torch.data(jpegs)))
	local offsets_p = ffi.cast('unsigned long *', ffi.cast('intptr_t', torch.data(d.offsets)))
	local sizes_p   = ffi.cast('unsigned long *', ffi.cast('intptr_t', torch.data(d.sizes)))
	local gm = require 'graphicsmagick'

	local n = d.labels:size(1)
	local dt = {}
	dt.data = torch.FloatTensor(n, 3, opt.height, opt.width)
	dt.labels = torch.Tensor(n)

	for i = 1, n do

		xlua.progress(i, n)

		local offset = tonumber(offsets_p[i-1] - 1)
		local size = tonumber(sizes_p[i-1])
		local jpegblob = jpegs_p + offset
		local sample = gm.Image():fromBlob(jpegblob,size):toTensor('float','RGB','DHW',true)
		dt.data[i] = image.scale(sample, opt.width, opt.height)	
		dt.labels[i] = d.labels[i]
		--image.display(sample)

	end

	dt.classes = d.classes or class_names
	return dt 
		
end

----------------------------------------------------------------------
function show_classes(data)

   local k = 100
   local ivch = data.data:size(2)
   local ivhe = data.data:size(3)
   local ivwi = data.data:size(4)

   local disp_ims = {}
	local nclasses = #data.classes

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
      image.display({ image = disp_ims[i], legend = data.classes[i], padding=4 })
   end

end


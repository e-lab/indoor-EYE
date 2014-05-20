function load_data_mm_multi(data_file, info_file)

   local dataset = {}
   local nbThread = 2

   -- (1) prepare shared data (as pointers)
   -- (1.1) shuffle tensor
   dataset.shuffle = torch.randperm(nsamples):type('torch.LongTensor')
   local shuffle_p = tonumber(torch.data(dataset.shuffle, true))

   -- (1.2) global mean and std
   local gm_p = tonumber(torch.data(global_mean, true))
   local gstd_p = tonumber(torch.data(global_std, true))

   -- (1.3) data destination (one for each thread)
   dataset.samples = {}
   dataset.targets = {}
   dataset.samples_p = torch.LongTensor(nbThread)
   dataset.targets_p = torch.LongTensor(nbThread)
   for th = 1, nbThread do
      dataset.samples[th] = torch.FloatTensor(opt.batchSize, opt.color, opt.height, opt.width)
      samples_p           = tonumber(torch.data(samples[th], true))
      dataset.targets[th] = torch.FloatTensor(opt.batchSize)
      targets_p           = tonumber(torch.data(targets[th], true))
   end

   local samples_pp = tonumber(torch.data(dataset.samples_p, true))
   local targets_pp = tonumber(torch.data(dataset.targets_p, true))

   -- (1.4) dataset information
   dataset.info = torch.load(info_file)
   local offsets_p     = torch.data(dataset.info.offsets)
   local sizes_p       = torch.data(dataset.info.sizes)
   local labels_p      = torch.data(dataset.info.labels)
   local file_number_p = torch.data(dataset.info.file_number)

   -- (1.5) images
   dataset.jpegs = {}
   local jpegs_p
   local nbFile = 1
   if (infoDataset.file_range) then
      nbFile = dataset.info.file_range:size(1)
      jpegs_p = torch.LongTensor(nbFile)
      for file = 1, nbFile do
         jpegs[file]   = torch.ByteStorage(data_file .. '-file' .. file .. '.t7')
         jpegs_p[file] = tonumber(torch.data(jpegs[file], true))
      end
   else
      jpegs_p = torch.LongTensor(nbFile)
      jpegs[1] = torch.ByteStorage(data_file)
      jpegs_p[1] = tonumber(torch.data(jpegs[1], true))
   end
   local jpegs_pp = tonumber(torch.data(jpegs_p, true))

   -- (2) prepare thread callback function
   local function requirements()
      ffi = require 'ffi'
      gm = require 'graphicmagick'
      mf = math.floor
   end

   local function initData()

      nSamples = dataset.info.labels:size(1)
      c,h,w = opt.color, opt.height, opt.width
      bs = opt.batchSize
      batchLength = bs*c*h*w

      -- (1) get shared data
      sharedShuffle_p = ffi.cast('unsigned long *', shuffle_p)
      sharedGlobalMean_p = ffi.cast('unsigned long *', gm_p)
      sharedGlobalStd_p = ffi.cast('unsigned long *', gstd_p)

      sharedSamples_pp = torch.LongTensor(torch.LongStorage(nbThread, samples_pp)):resize(nbThread)
      sharedTargets_pp = torch.LongTensor(torch.LongStorage(nbThread, targets_pp)):resize(nbThread)

      -- (2) prepare dataset
      -- (2.1) making ffi pointer to useful informations
      sharedOffsets_p     = ffi.cast('unsigned long *', offset_p)
      sharedSizes_p       = ffi.cast('unsigned long *', sizes_p)
      sharedLabels_p      = ffi.cast('unsigned long *', labels_p)
      sharedFile_number_p = ffi.cast('unsigned int *', file_number_p)

      -- (2.3) pointer to the ByteStorage to different file
      local sharedJpegs_pp = torch.LongTensor(torch.LongStorage(nbFile, jpegs_pp)):resize(nbFile)
      sharedJpegs_p = {}
      for (file = 1, nbFile) do
         sharedJpegs_p[file] = ffi.cast('unsigned char *', sharedJpegs_pp[file])
      end

   end

   dataset.jobToDo = function (threadId, idx, test)

      -- (1) get data destination
      sample = torch.FloatTensor(torch.FloatStorage(batchLength, sharedSamples_pp[threadId])):resize(bs, c, h, w)
      sample:zero()
      target_p = ffi.cast('unsigned float *', sharedTargets_p[threadId])

      -- (2) process batch:
      for i = 0,bs-1 do
         -- offsets:
         local start = ((idx-1)*bs + i) % nSamples
         local ii = sharedShuffle_p[start] - 1

         -- decode jpeg:
         local offset = tonumber(sharedOffsets_p[ii] - 1)
         local size = tonumber(sharedSizes_p[ii])
         local numFile = tonumber(sharedFile_number_p[ii])
         local jpegblob = sharedJpegs_p[numFile] + offset
         local sample = gm.Image():fromBlob(jpegblob,size):toTensor('float','RGB','DHW',true)

         -- extract square patch
         local size = math.min(sample:size(2), sample:size(3))
         local t = mf((sample:size(2) - size)/2 + 1)
         local l = mf((sample:size(3) - size)/2 + 1)
         local b = t + size - 1
         local r = l + size - 1
         sample = sample[{ {},{t,b},{l,r} }]

         sample = image.scale(sample, w + jitter, h + jitter)

         if jitter > 0 and not test then
            -- extract sub-patch, with optional jitter:
            size = math.min(sample:size(2), sample:size(3))
            t = mf((sample:size(2) - h)/2 + 1)
            l = mf((sample:size(3) - w)/2 + 1)

            t = t + mf(torch.uniform(-jitter/2,jitter/2))
            l = l + mf(torch.uniform(-jitter/2,jitter/2))

            b = t + h - 1
            r = l + w - 1
            sample = sample[{ {},{t,b},{l,r} }]
         end

         -- normalize sample
         for j = 1, c do
            sample[j]:add(-global_mean[j])
            sample[j]:div(global_std[j])
         end

         -- save sample:
         samples[i+1] = sample

         -- label:
         targets_p[i] = labels_p[ii]
      end

      return
   end

   dataset.finishJob = function finishJob()
   end

   -- (3) create threads
   dataset.threads = require('MyThreads')(nbThreadb, requirements, initData)

   -- (4) prepare main thread functions
   dataset.nBatches = math.floor(nsamples / bs)
   dataset.threadId = torch.IntTensor(dataset.nBatches)

   dataset.preparebatch = function(t, test)
      dataset.threadId[t] = dataset.threads:addjob(dataset.jobToDo, dataset.finishJob, t, test)
   end

   dataset.copyBatch = function(t, ims, tgs)
      if (dataset.threadId[t] > 0) then
         dataset.threads:synchronize(dataset.threadId[t])
         ims:copy(dataset.samples[dataset.threadId[t]])
         tgs:copy(dataset.targets[dataset.threadId[t]])
         dataset.threadId[t] = 0
      else
         error(string.format("Thread %d hasn't be prepared.", t))
      end
   end

   dataset.newShuffle = function()
      dataset.shuffle:copy(torch.randperm(nsamples):type('torch.LongTensor'))
   end


   return dataset
end

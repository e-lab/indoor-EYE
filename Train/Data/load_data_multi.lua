function load_data_mm_multi(data_file, info_file)

   local dataset = {}

   -- (1) get the opt values...
   local nbThreads = opt.loading_thread
   local height = opt.height
   local width = opt.width
   local batchSize = opt.batchSize
   local jitter = opt.jitter
   local batchLength = batchSize*3*height*width

   -- (1) prepare shared data (as pointers)
   -- (1.1) dataset information
   dataset.info = torch.load(info_file)
   local nSamples = dataset.info.labels:size(1)
   local offsets_p     = torch.data(dataset.info.offsets, true)
   local sizes_p       = torch.data(dataset.info.sizes, true)
   local labels_p      = torch.data(dataset.info.labels, true)
   local file_number_p = torch.data(dataset.info.file_number, true)

   -- (1.2) shuffle tensor
   dataset.shuffle = torch.randperm(nSamples):type('torch.LongTensor')
   local shuffle_p = tonumber(torch.data(dataset.shuffle, true))

   -- (1.3) global mean and std
   local gm_p = tonumber(torch.data(global_mean, true))
   local gstd_p = tonumber(torch.data(global_std, true))

   -- (1.4) data destination (one for each thread)
   dataset.samples = {}
   dataset.targets = {}
   local samples_p = torch.LongTensor(nbThreads)
   local targets_p = torch.LongTensor(nbThreads)
   for th = 1, nbThreads do
      dataset.samples[th] = torch.FloatTensor(opt.batchSize, 3, opt.height, opt.width)
      samples_p[th]       = tonumber(torch.data(dataset.samples[th], true))
      dataset.targets[th] = torch.LongTensor(opt.batchSize)
      targets_p[th]       = tonumber(torch.data(dataset.targets[th], true))
   end

   local samples_pp = tonumber(torch.data(samples_p, true))
   local targets_pp = tonumber(torch.data(targets_p, true))

   -- (1.5) images
   dataset.jpegs = {}
   local jpegs_p
   local nbFile = 1
   if (dataset.info.file_range) then
      nbFile = dataset.info.file_range:size(1)
      jpegs_p = torch.LongTensor(nbFile)
      for file = 1, nbFile do
         dataset.jpegs[file]   = torch.ByteStorage(data_file .. '-file' .. file .. '.t7')
         jpegs_p[file] = tonumber(torch.data(dataset.jpegs[file], true))
      end
   else
      jpegs_p = torch.LongTensor(nbFile)
      dataset.jpegs[1] = torch.ByteStorage(data_file)
      jpegs_p[1] = tonumber(torch.data(dataset.jpegs[1], true))
   end
   local jpegs_pp = tonumber(torch.data(jpegs_p, true))

   --  (1) threads init functions
   local function requirements ()
      ffi = require 'ffi'
      gm = require 'graphicsmagick'
      mf = math.floor
      require 'image'
   end
   -- initData
   local function initData()

      c,h,w = 3, height, width
      bs = batchSize
      -- (1) get shared data
      sharedShuffle_p = ffi.cast('unsigned long *', shuffle_p)
      sharedGlobalMean_p = ffi.cast('float *', gm_p)
      sharedGlobalStd_p = ffi.cast('float *', gstd_p)

      sharedSamples_pp = torch.LongTensor(torch.LongStorage(nbThreads, samples_pp)):resize(nbThreads)
      sharedTargets_pp = torch.LongTensor(torch.LongStorage(nbThreads, targets_pp)):resize(nbThreads)

      -- (2) prepare dataset
      -- (2.1) making ffi pointer to useful informations
      sharedOffsets_p     = ffi.cast('unsigned long *', offsets_p)
      sharedSizes_p       = ffi.cast('unsigned long *', sizes_p)
      sharedLabels_p      = ffi.cast('unsigned long *', labels_p)
      sharedFile_number_p = ffi.cast('unsigned int *', file_number_p)

      -- (2.3) pointer to the ByteStorage to different files
      local sharedJpegs_pp = torch.LongTensor(torch.LongStorage(nbFile, jpegs_pp)):resize(nbFile)
      sharedJpegs_p = {}
      for file = 1, nbFile do
         sharedJpegs_p[file] = ffi.cast('unsigned char *', sharedJpegs_pp[file])
      end

      -- (3) prepare data destination
      -- samples = torch.FloatTensor(bs, c, h, w)
      local sharedSamples_pp = torch.LongTensor(torch.LongStorage(nbThreads, samples_pp)):resize(nbThreads)
      threadSamples = torch.FloatTensor(torch.FloatStorage(batchLength, sharedSamples_pp[threadId])):resize(bs, c, h, w)
      threadSamples:zero()
      -- targets = torch.LongTensor(bs)
      local sharedTargets_pp = torch.LongTensor(torch.LongStorage(nbThreads, targets_pp)):resize(nbThreads)
      threadTargets_p = ffi.cast('unsigned long *', sharedTargets_pp[threadId])

   end

   -- (2) prepare job function
   dataset.jobToDo = function (tId, idx, test)
      assert(tId == threadId)

      threadSamples:zero()

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
            sample[j]:add(-sharedGlobalMean_p[j-1])
            sample[j]:div(sharedGlobalStd_p[j-1])
         end

         -- save sample:
         threadSamples[i+1] = sample

         -- label:
         threadTargets_p[i] = sharedLabels_p[ii]
      end

      return threadId
   end

   dataset.finishJob = function(tId)
   end
   -- (3) create threads
   dataset.threads = require('Data/MyThreads')(nbThreads, requirements, initData)

   -- (4) prepare main thread functions
   dataset.nBatches = math.floor(nSamples / batchSize)
   dataset.threadId = torch.IntTensor(dataset.nBatches)

   dataset.prepareBatch = function(t, test)
      dataset.threadId[t] = dataset.threads:addjob(dataset.jobToDo, dataset.finishJob, t, test)
   end

   dataset.copyBatch = function(t, ims, tgs)
      if (dataset.threadId[t] > 0) then
         dataset.threads:synchronize(dataset.threadId[t])
         ims:copy(dataset.samples[dataset.threadId[t]])
         tgs:copy(dataset.targets[dataset.threadId[t]])
         dataset.threadId[t] = 0
      else
         error(string.format("The batch %d hasn't be prepared.", t))
      end
   end

   dataset.newShuffle = function()
      dataset.shuffle:copy(torch.randperm(nSamples):type('torch.LongTensor'))
   end


   return dataset
end

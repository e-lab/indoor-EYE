-------------------------------------------------------------------------------
-- Data processing functions
-- Artem Kuharenko
-------------------------------------------------------------------------------

function print_sizes(data, text, verbose)
--print dimensions of data

   if verbose then
      print('==> ' .. text .. ' number of batches: ' .. data.nbatches() .. ', Batch size: ' .. opt.batchSize .. 'x 3 x' .. opt.height .. 'x' .. opt.width)
   end

end

function verify_statistics(images, channels, text, v)

   if v then

      print('==> verify ' .. text .. ' statistics:')

      for i, channel in ipairs(channels) do

         local mean = images[{ {}, {i}, {}, {} }]:mean()
         local std = images[{ {}, {i}, {}, {} }]:std()

         print('       ' .. channel .. '-channel, mean:                   ' .. mean)
         print('       ' .. channel .. '-channel, standard deviation:     ' .. std)

      end

   end

end

function get_global_mean(data, verbose)

   local nc = data.data:size(2)
   local mean = torch.Tensor(nc):zero()
   local std = torch.Tensor(nc):zero()

   for i = 1, nc do

      mean[i] = data.data[{ {},i,{},{} }]:mean()
      std[i] = data.data[{ {},i,{},{} }]:std()

   end

   if verbose then
      print('Mean: ' .. mean[1] .. ' ' .. mean[2] .. ' ' .. mean[3])
      print('Std: ' .. std[1] .. ' ' .. std[2] .. ' ' .. std[3])
   end

   return mean, std

end

-- with the whole dataset give the whole name
-- with the dataset splited, for data_file, only give until the subsample_name:
-- <folder>/train-data-imagenet (the files are train-data-imagenet-file*.t7)
function get_global_mean_async(data_file, info_file, sdir, save_act, verbose)

   local mean = torch.Tensor(opt.ncolors):zero()
   local std = torch.Tensor(opt.ncolors):zero()
   local sfile = sdir .. 'preproc.t7'

   if save_act == 'load' and paths.filep(sfile) then

      if verbose then
         print('==> Loading mean and std from file ' .. sfile)
      end
      local preproc = torch.load(sfile)
      mean = preproc.mean
      std = preproc.std

   else

      if verbose then
         print('==> Computing global mean and std')
      end

      -- get info table
      local dt = torch.load(info_file)
      local total_size = dt.labels:size(1)
      local offsets_p = ffi.cast('unsigned long *', ffi.cast('intptr_t', torch.data(dt.offsets)))
      local sizes_p   = ffi.cast('unsigned long *', ffi.cast('intptr_t', torch.data(dt.sizes)))
      local gm = require 'graphicsmagick'

      -- index_global is the index of the current image in the info table
      local index_global = 1

      -- get the number of file (if file_range doesn't exist, then it is the whole dataset)
      local number_file = 1
      if (dt.file_range) then
         number_file = dt.file_range:size(1)
      end

      if verbose then
         print('===> ' .. number_file .. ' files')
      end

      for file = 1, number_file do
         -- verifying if we have the good number
         assert(index_global == dt.file_range[file][1])

         if verbose then
            print('===> file #' .. file)
         end

         -- get the storage of the file
         local jpegs
         if (dt.file_range) then
            jpegs = torch.ByteStorage(data_file .. '-file' .. file .. '.t7')
         else
            jpegs = torch.ByteStorage(data_file)
         end

         local jpegs_p   = ffi.cast('unsigned char *', ffi.cast('intptr_t', torch.data(jpegs)))

         -- get the number of image in this file
         local number_image = total_size
         if (dt.file_range) then
            number_image = dt.file_range[file][2] - dt.file_range[file][1] + 1
         end

         for i = 1, number_image do

            if verbose and (i == 1 or i%100 == 0 or i == number_image) then
               xlua.progress(i, number_image)
            end

            local offset = tonumber(offsets_p[index_global-1] - 1)
            local size = tonumber(sizes_p[index_global-1])
            local jpegblob = jpegs_p + offset
            local sample = gm.Image():fromBlob(jpegblob,size):toTensor('float','RGB','DHW',true)
            sample = extract_square_patch(sample)
            sample = image.scale(sample, opt.width, opt.height)

            for j = 1, opt.ncolors do

               mean[j] = mean[j] + sample[j]:mean() / total_size
               local s2 = sample[j]:dot(sample[j])
               std[j] = std[j] + s2 / total_size / sample:size(2) / sample:size(3)

            end
            index_global = index_global + 1
         end
         jpegs = nil
         collectgarbage()
      end

      for j = 1, opt.ncolors do
         std[j] = std[j] - mean[j] * mean[j]
      end
      std:sqrt()

      if save_act == 'save' then
         local preproc = {}
         preproc.mean = mean
         preproc.std = std
         torch.save(sfile, preproc)
      end

   end

   if verbose then
      print('======> Using train data mean: ' .. mean[1] .. ' ' .. mean[2] .. ' ' .. mean[3])
      print('======> Using train data std: ' .. std[1] .. ' ' .. std[2] .. ' ' .. std[3])
   end

   return mean, std

end



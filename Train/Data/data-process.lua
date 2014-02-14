-------------------------------------------------------------------------------
-- Data processing functions
-- Artem Kuharenko
-------------------------------------------------------------------------------

function print_sizes(data, text, verbose)
--print dimensions of data

   data.prepareBatch(1)
   local ims = data.copyBatch()

   if verbose then
      print('==> ' .. text .. ' number of batches: ' .. data.nbatches() .. ', Batch size: ' .. ims:size(1) .. 'x' .. ims:size(2) .. 'x' .. ims:size(3) .. 'x' .. ims:size(4))
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

      local dt = torch.load(info_file)
      local jpegs = torch.ByteStorage(data_file)

      local jpegs_p   = ffi.cast('unsigned char *', ffi.cast('intptr_t', torch.data(jpegs)))
      local offsets_p = ffi.cast('unsigned long *', ffi.cast('intptr_t', torch.data(dt.offsets)))
      local sizes_p   = ffi.cast('unsigned long *', ffi.cast('intptr_t', torch.data(dt.sizes)))
      local gm = require 'graphicsmagick'

      local n = dt.labels:size(1)

      for i = 1, n do

         if verbose and i*100/n%10 == 0 then
            xlua.progress(i, n)
         end

         local offset = tonumber(offsets_p[i-1] - 1)
         local size = tonumber(sizes_p[i-1])
         local jpegblob = jpegs_p + offset
         local sample = gm.Image():fromBlob(jpegblob,size):toTensor('float','RGB','DHW',true)
         sample = extract_square_patch(sample)
         sample = image.scale(sample, opt.width, opt.height)

         for j = 1, opt.ncolors do

            mean[j] = mean[j] + sample[j]:mean() / n
            local s2 = sample[j]:dot(sample[j])
            std[j] = std[j] + s2 / n / sample:size(2) / sample:size(3)

         end

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



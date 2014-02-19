------------------------------------------------------------
-- Alfredo Canziani, Artem Kuharenko
-- original code and net training by Clement Farabet
--
------------------------------------------------------------

lapp = require 'pl.lapp'
lapp.slack = true
require 'qt'
require 'qtwidget'
require 'ffmpeg'
require 'imgraph' -- to colorize outputs
--require 'segmtools' -- to display name of classes
require 'nnx'
require 'image'
--require 'camera'
--nn_X = dofile('../src/nn_X.lua')--require 'nn_X'

----------------------------------------------------------------------
print '==> processing options'

opt = lapp[[
  -x,   --runnnx        (default false)       run on hardware nn_X 
  -t,   --threads       (default 3)          number of threads
  -v,   --video         (default '')         video (or image) file to process
  -n,   --network       (default 'multinet-float-ascii.net') path to trained network
  		  --networktype   (default 'cnn')      type of network ('cnn' or 'unsup')
  -s,   --save          (default '')         path to save output video
  -l,   --useffmpeglib  (default false)      help=use ffmpeglib module to read frames directly from video
  -k,   --seek          (default 0)          seek number of seconds
  -f,   --fps           (default 10)         number of frames per second
        --seconds       (default 10)         length to process (in seconds)
  -w,   --width         (default 320)        resize video, width 
  -h,   --height        (default 200)        resize video, height
  -z,   --zoom          (default 1)          display zoom
        --downsampling  (default 2)          downsample input frame for processing
  -c,   --camidx        (default 0)          if source=camera, specify the camera index: /dev/videoIDX
]]

opt.runnnx = false
opt.downsampling = tonumber(opt.downsampling)

torch.setdefaulttensortype('torch.FloatTensor')
torch.setnumthreads(opt.threads)

-- type:
if opt.runnnx then
   print('==> switching to nn_X hardware platform')
end

---------------------------------------------------------------------------------
--functions to convert cuda model to float model
--see https://github.com/Atcold/Torch7-tools/blob/master/netConverter.lua

function smartCopy(cudaModule,floatNetwork)
   -- if cudaModule.__typename == 'nn.Sequential' then
   --    floatNetwork = nn.Sequential()
   if cudaModule.__typename == 'nn.SpatialConvolutionCUDA' then
      print(' + Converting <nn.SpatialConvolutionCUDA> into <nn.SpatialConvolution>')
      floatNetwork:add(nn.SpatialConvolution(cudaModule.nInputPlane, cudaModule.nOutputPlane, cudaModule.kW, cudaModule.kH))
      floatNetwork.modules[#floatNetwork.modules].gradBias   = nil
      floatNetwork.modules[#floatNetwork.modules].gradWeight = nil
      floatNetwork.modules[#floatNetwork.modules].gradInput  = nil
      floatNetwork.modules[#floatNetwork.modules].weight     = cudaModule.weight:transpose(4,3):transpose(3,2):transpose(2,1):float()
      floatNetwork.modules[#floatNetwork.modules].bias       = cudaModule.bias:float()
   elseif cudaModule.__typename == 'nn.SpatialMaxPoolingCUDA' then
      print(' + Converting <nn.SpatialMaxPoolingCUDA> into <nn.SpatialMaxPooling>')
      floatNetwork:add(nn.SpatialMaxPooling(cudaModule.kW, cudaModule.kH, cudaModule.dW, cudaModule.dH))
      --floatNetwork.modules[#floatNetwork.modules].indices    = nil
      floatNetwork.modules[#floatNetwork.modules].gradInput  = nil
   elseif cudaModule.__typename ~= 'nn.Transpose' then
      print(' + Copying <' .. cudaModule.__typename .. '>')
      floatNetwork:add(cudaModule)
   end
end

function convert(cudaNetwork)
   local floatNetwork = nn.Sequential()
   if cudaNetwork.modules then
      for _,a in ipairs(cudaNetwork.modules) do
         smartCopy(a,floatNetwork)
      end
   end
   return floatNetwork
end
---------------------------------------------------------------------------------

---------------------------------------------------------------------------------
--Define neuralnet functions
---------------------------------------------------------------------------------
function init_net()

   local data = {}
   print('loading unsup model')
   data.preproc = torch.load('../Train/results/preproc.t7')
   local model = torch.load('../Train/results/model-5.net')

   local cnn = model:get(1)
   data.cnn = convert(cnn)
   data.classifier = model:get(2)
   --print(data.u1net.modules)
   --print(data.classifier.modules)
   
   local classes = {'mouse', 'printer', 'cellphone', 'cup', 'laptop', 'keyboard', 
                    'desk', 'bottle-of-water', 'trash-can'}

   local colours = {[ 1] = {0.7, 0.7, 0.3}, -- mouse
				        [ 2] = {0.4, 0.4, 0.8}, -- printer
				        [ 3] = {0.0, 0.9, 0.0}, -- phone
				        [ 4] = {1.0, 0.0, 0.3}, -- cup
				        [ 5] = {0.3, 0.3, 0.3}, -- laptop
				        [ 6] = {1.0, 0.1, 0.1}, -- keyboard
				        [ 7] = {0.0, 0.7, 0.9}, -- desk
				        [ 8] = {0.0, 0.0, 1.0}, -- bottle
				        [ 9] = {0.5, 0.5, 0.0}} -- trashcan
			
   return data, classes, colours

end

function preprocess_frame(alg_data, frame)

   for i = 1,3 do
      frame[{ {i},{},{} }]:add(-alg_data.preproc.mean[i])
      frame[{ {i},{},{} }]:div(alg_data.preproc.std[i])
   end

   return frame

end

function get_features(alg_data, frame)

	features = alg_data.cnn:forward(frame)
   print(#features)
   return features

end


function get_distributions(alg_data, features)

	d = 12

	stride = 12
	nfy = math.floor(features:size(2) / stride) 
	nfx = math.floor(features:size(3) / stride) 
   
	local distributions = torch.Tensor(#classes, nfy, nfx)
	local f_list = torch.Tensor(nfx * nfy, features:size(1), d, d)

	local i = 0
	--reoder features
   for y = 1, nfy do
      for x = 1, nfx do
			
			i = i + 1
			x1 = (x - 1) * stride + 1
			x2 = x1 + d - 1
			y1 = (y - 1) * stride + 1
			y2 = y1 + d - 1
			f_list[i] = features[{{},{y1, y2}, {x1, x2}}] -- if stride >= d then we could just copy references

      end
   end

   print(#f_list)
	--compute distributions
	local d_list = alg_data.classifier:forward(f_list) 
   
   print(#d_list)
	--reoder distributions
	i = 0
   for y = 1, nfy do
      for x = 1, nfx do

			i = i + 1
			distributions[{{}, {y}, {x}}] = d_list[i]

		end
	end

   return distributions

end
---------------------------------------------------------------------------------

alg_data, classes, colours = init_net()
print(alg_data)
-- generating the <colourmap> out of the <colours> table
colormap = imgraph.colormap(colours)

-- load video
if opt.video ~= '' then
   video = ffmpeg.Video{path=opt.video,
   width=opt.width, height=opt.height,
   fps=opt.fps, length=opt.seconds, seek=opt.seek,
   encoding='jpg',
   delete=false}
else
   camera = image.Camera{}
end

-- setup GUI (external UI file)
if not win or not widget then
   win = qtwidget.newwindow(opt.width*opt.zoom, opt.height*opt.zoom*2 + 80, -- 20 for class display
   'E-Lab RoadNet demo')
   font = qt.QFont{serif=false, italic=false, size=12}
   win:setfont(font)
end

-- allocate the input
local ivch = 3
local iH = opt.height/opt.downsampling
local iW = opt.width/opt.downsampling
local img_temp = torch.Tensor(ivch, iH, iW)
if opt.runnnx then
   nn_X:input(img_temp)
   nb_outs, oH,oW, ops = nn_X:compile(network, ivch, iH, iW)
end

local hwt = 0

-- process and time in SW on CPU:
cput = 0
for i = 1, 10 do
	sys.tic() --test on HW
	features = get_features(alg_data, img_temp)
	cput = cput + sys.toc()
end
cput = cput / 10
print('CPU frame precessing time [ms]: ', cput*1000)


-- process function
function process()
   -- grab frame
   if opt.video ~= '' then
      fframe = video:forward()
   else
      fframe = camera:forward()
   end

   local width = opt.width
   local height = opt.height

   cframe = fframe
   if opt.downsampling > 1 then
      width  = width/opt.downsampling
      height = height/opt.downsampling
      frame = image.scale(cframe, width, height)
   else
      frame = cframe:clone()
   end

	frame = preprocess_frame(alg_data, frame)

   -- process frame with network:  
   if opt.runnnx then
      print('Process')
      sys.tic()
      nn_X:forward()
      hwt = sys.toc()
      if not intm1 then -- we do not want to do this in the loop, just once! 
         features = torch.Tensor(nb_outs, nn_X.outs[#nn_X.outs][1]:size(1), nn_X.outs[#nn_X.outs][1]:size(2))
      end
      for i=1, nb_outs do
         features[i] = nn_X.outs[#nn_X.outs][i]:clone()
      end
   else
		sys.tic()
      features = get_features(alg_data, frame)
		print('features ' .. sys.toc())   
	end
  
   -- (a) compute class distributions
	sys.tic()
	distributions = get_distributions(alg_data, features)
	print('distributions ' .. sys.toc())   

   -- (b) upsample the distributions
   distributions = image.scale(distributions, frame:size(3), frame:size(2), 'simple')

   -- (d) winner take all
   _,winners = torch.max(distributions,1)
   winner = winners[1]
end


-- display function
function display()
   -- colorize classes
   colored, colormap = imgraph.colorize(winner, colormap)

   -- display raw input
   image.display{image=fframe, win=win, zoom=opt.zoom, min=0, max=1}

   -- map just the processed part back into the whole image
   if opt.downsampling > 1 then
      colored = image.scale(colored,fframe:size(3),fframe:size(2))
   end
   colored:add(fframe)
   
   -- overlay segmentation on input frame
   image.display{image=colored, win=win, y=fframe:size(2)*opt.zoom, zoom=opt.zoom, min=0, max=colored:max()}

   -- print classes:
   -- this if you want to print predefine color labels ffclasses:
   --image.display{image=ffclasses, win=win, y=2*fframe:size(2)*opt.zoom, zoom=opt.zoom, min=0, max=1}
   for i = 1,#classes do
      local dx = 52
      local x = (i-1)*dx
      win:rectangle(x, opt.height*opt.zoom*2, dx, 20)--opt.height*opt.zoom*2+20)
      win:setcolor(colours[i][1],colours[i][2],colours[i][3])
      win:fill()
      win:setcolor('black')--colours[i])
      win:moveto(x+5, opt.height*opt.zoom*2 + 15)
      win:show(classes[i])
   end

   -- display profile data:
   local speedup = cput/hwt
   str1 = string.format('CPU compute time [ms]: %f',  cput*1000)
   str2 = string.format('HW config+compute time [ms]: %f', hwt*1000)
   str3 = string.format('speedup = %f ',  speedup)
   win:setfont(qt.QFont{serif=false,italic=false,size=12})
   -- disp line:
   win:moveto(10, opt.height*opt.zoom*2 + 35);
   win:show(str1)
   win:moveto(10, opt.height*opt.zoom*2 + 55);
   win:show(str2)
   win:moveto(10, opt.height*opt.zoom*2 + 75);
   win:show(str3)
   
   -- save ?
   if opt.save ~= '' then
      local t = win:image():toTensor(3)
      local fname = opt.save .. string.format('/frame_%05d.jpg',times)
      sys.execute(string.format('mkdir -p %s',opt.save))
      print('saving:'..fname)
      image.save(fname,t)
   end
end

times = 0
-- setup gui
while win:valid() do
   process()
   times = times + 1
   win:gbegin()
   win:showpage()
   display()
   win:gend()
   collectgarbage()
end

-- exit
if xdma.exit() < 0 then
   print("ERROR: Could not exit xdma_torch library")
   exit()
end

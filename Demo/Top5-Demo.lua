--------------------------------------------------------------------------------
-- Top5Demo
--------------------------------------------------------------------------------
-- Displays the top 5 guesses of the network for the current image.
-- Current image can be drawn from the testing dataset or from camera.
--------------------------------------------------------------------------------

-- Requires --------------------------------------------------------------------
require 'pl'
require 'qtwidget'
require 'nnx'
require 'camera'
require 'sys'
require 'gnuplot'

-- Title definition -----------------------------------------------------------
local title = [[
           _______         _____       _____
          |__   __|       | ____|     |  __ \
             | | ___  _ __| |__ ______| |  | | ___ _ __ ___   ___
             | |/ _ \| '_ \___ \______| |  | |/ _ \ '_ ` _ \ / _ \
             | | (_) | |_) |__) |     | |__| |  __/ | | | | | (_) |
             |_|\___/| .__/____/      |_____/ \___|_| |_| |_|\___/
                     | |
                     |_|

]]

-- Options ---------------------------------------------------------------------
opt = lapp(title .. [[
--temp_dir       (default '../Train/temp-data/') Location of test dataset
--camera                                         Switch to camera input
--subsample_name (default 'indoor51')            Name of imagenet subsample. Possible options ('class51', 'elab')
--model          (default '07C/model-90.net'   ) Model's name
--histogram                                      Shows prediction's histogram
--imageSide      (default 128                  ) Image's side length
--fps            (default 2                    ) Frames per second (camera setting)
--hdcam          (default true                 ) Use of HD mac camera for demos
]])
io.write(title)
torch.setdefaulttensortype('torch.FloatTensor')

-- Loading different parts -----------------------------------------------------
-- Loading net
netPath = opt.model
net = torch.load(netPath)

-- Loading input
local stat
if opt.camera then
   cam = image.Camera{}
   local statPath = string.gsub(netPath,'%a+%-%d+%.net','preproc.t7')
   stat = torch.load(statPath)
else
   local f = opt.temp_dir .. 'test.t7'
   testData = torch.load(f)
end

-- Loading classes' names
dofile('../Train/Data/indoor-classes.lua')

-- Build window (not final solution)
win = qtwidget.newwindow(4*opt.imageSide,4*opt.imageSide,'New window')

-- Displaying routine
function show(idx)
   local input, l, c, leg
   if opt.camera then
      if opt.hdcam then input = image.scale(cam:forward(), opt.imageSide*16/9,  opt.imageSide)
      else input = image.scale(cam:forward(),'^' .. opt.imageSide)
      end
      local w = (#input)[3]
      input = image.crop(input,w/2-opt.imageSide/2,0,w/2+opt.imageSide/2,opt.imageSide)
      for c = 1,3 do
         input[c]:add(-stat.mean[c])
         input[c]:div( stat.std [c])
      end
   else
      l = testData.labels[idx] -- true label nb
      c = classes[l][1] -- true label str
      leg = l .. ': ' .. c -- legend
      if opt.histogram then gnuplot.title(leg) end
      win.widget:setWindowTitle('True label: ' .. c)
      input = testData.data[idx]
   end

   -- Computing prediction
   local output = net:forward(input)

   -- Displaying current frame
   image.display{image = input, zoom = 4, win = win}

   -- Show histogram, if required
   if opt.histogram then
      gnuplot.bar(output)
      gnuplot.grid('on')
      gnuplot.axis({0,52,-15,0})
   end

   -- Drawing semi-transparent rectagle on top left
   win:rectangle(0,0,180,110)
   win:setcolor(0,0,0,.3)
   win:fill()

   -- Set font size to a visible dimension
   win:setfontsize(20)

   -- Computing index of decreasing value ordered prob.
   local sortedOutput, guess = output:sort(true)

   -- Printing first 5 most likely predictions
   for i = 1,5 do
      win:rectangle(10,3+20 * i,math.exp(sortedOutput[i])*130,2)
      win:moveto(10,20 * i)
      if guess[i] == l and not opt.camera then
         win:setcolor('red')
      else
         win:setcolor('white')
      end
      win:fill()
      win:show(classes[guess[i]][1])
   end
end

-- While loop: running program
if not opt.camera then
   print('Press <Ctrl> + D for switching to the next image, <Ctrl> + C + <Return> for exiting')
end

i = 1
while win:valid() do
   show(i)
   i = i+1
   if not opt.camera then
      io.read()
   else
      sys.sleep(1/opt.fps)
   end
end

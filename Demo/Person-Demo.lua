--------------------------------------------------------------------------------
-- Person-Demo
--------------------------------------------------------------------------------
-- Demo for INRIA dataset person detector
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
   _____                                   _      _            _
  |  __ \                                 | |    | |          | |
  | |__) |__ _ __ ___  ___  _ __ ______ __| | ___| |_ ___  ___| |_ ___  _ __
  |  ___/ _ \ '__/ __|/ _ \| '_ \______/ _` |/ _ \ __/ _ \/ __| __/ _ \| '__|
  | |  |  __/ |  \__ \ (_) | | | |    | (_| |  __/ ||  __/ (__| || (_) | |
  |_|   \___|_|  |___/\___/|_| |_|     \__,_|\___|\__\___|\___|\__\___/|_|

]]

-- Options ---------------------------------------------------------------------
opt = lapp(title .. [[
--temp_dir       (default '../Train/temp-data/') Location of test dataset
--subsample_name (default 'indoor51')            Name of imagenet subsample. Possible options ('class51', 'elab')
--model          (string            )            Model's name
--histogram                                      Shows prediction's histogram
--imageSide      (default 128                  ) Image's side length
--camera                                         Switch to camera input
--camRes         (default VGA)                   Camera resolution ([VGA]|FWVGA|HD|FHD)
--fps            (default 20                   ) Frames per second (camera setting)
]])
io.write(title)
torch.setdefaulttensortype('torch.FloatTensor')

-- Loading different parts -----------------------------------------------------
-- Loading net
netPath = opt.model
-- netPath = '/home/atcold/work/indoor-EYE-results/' .. opt.model
-- netPath = '/Users/atcold/Work/Vision/DeconvNet/SW07C/' .. opt.model
net = torch.load(netPath, 'ascii')

-- Iterative function definition for disabling the dropouts
local function disableDropout(module)
   if module.__typename == 'nn.Dropout' then
      module.train = false
   end
end

local function findAndDisableDropouts(network)
   disableDropout(network)
   if network.modules then
      for _,a in ipairs(network.modules) do
         findAndDisableDropouts(a)
      end
   end
end

-- Disabling the dropouts
findAndDisableDropouts(net)

-- Loading input
local stat
local resolutions = {
   VGA   = {w =  640, h =  480},
   FWVGA = {w =  854, h =  480},
   HD    = {w = 1280, h =  720},
   FHD   = {w = 1920, h = 1080},
}

if opt.camera then
   cam = image.Camera{width = resolutions[opt.camRes].w, height = resolutions[opt.camRes].h}
   local statPath = string.gsub(netPath,'%a+%-%d+%.net.ascii','preproc.t7.ascii')
   stat = torch.load(statPath, 'ascii')
else
   local f = opt.temp_dir .. 'test.t7'
   testData = torch.load(f)
end

-- Loading classes' names
classes = {{'Torso',{1}},{'Body',{2}},{'Bckg',{3}}} -- for INRIA
cheatClasses = {'Person', 'Background'}

-- Build window (not final solution)
if opt.camera then
   win = qtwidget.newwindow(resolutions[opt.camRes].w,resolutions[opt.camRes].h,'TeraDeep Image Parser')
else
   win = qtwidget.newwindow(4*opt.imageSide,4*opt.imageSide,'TeraDeep Image Parser')
end

-- Displaying routine
eye = resolutions[opt.camRes].h * 3 / 4
x1  = resolutions[opt.camRes].w / 2 - eye / 2
y1  = resolutions[opt.camRes].h / 2 - eye / 2
x2  = resolutions[opt.camRes].w / 2 + eye / 2
y2  = resolutions[opt.camRes].h / 2 + eye / 2
kSize = 21
k   = torch.Tensor(kSize,kSize):fill(1/kSize^2)
z   = opt.camera and eye / 128 / 4 or 1 -- zoom

-- Set font size to a visible dimension
win:setfontsize(20*z)


-- profiling timers
timer = torch.Timer()
t_loop = 1

function show(idx)
   local input, l, c, leg, crop
   if opt.camera then
      frame = cam:forward()
      crop = image.crop(frame, x1, y1, x2, y2)
      input = image.scale(crop,'^' .. opt.imageSide)
      local w = (#input)[3]
      input = image.crop(input,w/2-opt.imageSide/2,0,w/2+opt.imageSide/2,opt.imageSide)
      for c = 1,3 do
         input[c]:add(-stat.mean[c])
         input[c]:div( stat.std [c])
      end
      frame = image.convolve(frame,k,'same')
      frame[{ {},{y1,y2-1},{x1,x2-1} }] = crop
   else
      l = testData.labels[idx] -- true label nb
      c = classes[l][1] -- true label str
      leg = l .. ': ' .. c -- legend
      if opt.histogram then gnuplot.title(leg) end
      win.widget:setWindowTitle('True label: ' .. c)
      input = testData.data[idx]
      frame = torch.Tensor(3,4*opt.imageSide,4*opt.imageSide)
      image.scale(frame, input)
   end

   -- Computing prediction
   local output = net:forward(input)

   -- Displaying current frame
   image.display{image = frame, win = win}

   -- Show histogram, if required
   if opt.histogram then
      gnuplot.bar(output)
      gnuplot.grid('on')
      gnuplot.axis({0,52,-15,0})
   end

   -- Drawing semi-transparent rectagle on top left
   if opt.camera then
      win:rectangle(0,0,x1,resolutions[opt.camRes].h)
      win:rectangle(x2,0,resolutions[opt.camRes].w,resolutions[opt.camRes].h)
      win:rectangle(x1,0,eye,y1)
      win:rectangle(x1,y2,eye,resolutions[opt.camRes].h)
   else
      win:rectangle(0,0,190,140)
   end
   win:setcolor(0,0,0,.4)
   win:fill()

   -- If cam, draw eye and write some text
   if opt.camera then
      win:moveto(30*z,45*z)
      win:setcolor(.1,.7,1)
      loopStr = 'Loop fps: ' .. 1/t_loop
      win:show(loopStr)
   end

   -- Computing conditional probability
   c = opt.camera and z or 0
   p = {}
   p[1] = output[1] + output[2]
   p[2] = output[3]

   -- Visualising
   for i = 1, 2 do
      win:rectangle(10*z+20*c,60*c+(4+25 * i)*z,p[i]*(190-20)*z,2*z)
      win:moveto(10*z+20*c,60*c+25 * i*z)
      if p[i] > .5 then
         win:setcolor(.3,.8,.3)
      else
         win:setcolor('white')
      end
      win:fill()
      predictionStr = string.format('%s (%.2f%%)', cheatClasses[i], p[i]*100)
      win:show(predictionStr)
   end

end

-- While loop: running program
if not opt.camera then
   print('Press <Ctrl> + D for switching to the next image, <Ctrl> + C + <Return> for exiting')
end

i = 1
while win:valid() do
   timer:reset()
   show(i)
   t_loop = timer:time().real
   i = i+1
   if not opt.camera then
      io.read()
   end
   collectgarbage()
end

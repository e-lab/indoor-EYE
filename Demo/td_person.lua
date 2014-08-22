--------------------------------------------------------------------------------
-- Person-Demo
--------------------------------------------------------------------------------
-- Demo for INRIA dataset person detector
--------------------------------------------------------------------------------

-- Requires --------------------------------------------------------------------
require 'pl'
require 'nnx'
require 'camera'
--require 'sys'

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
--model          (default 'model-125.net.ascii') Model's name
--imageSide      (default 149                  ) Image's side length
--camRes         (default VGA)                   Camera resolution ([VGA]|FWVGA|HD|FHD)
--fps            (default 10                   ) Frames per second (camera setting)
--pdt            (default 0.5)                   Detection threshold to detect person vs background
]])

io.write(title)
torch.setdefaulttensortype('torch.FloatTensor')

-- Loading net
local netPath = opt.model
local net = torch.load(netPath, 'ascii')

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

local cam = image.Camera{width = resolutions[opt.camRes].w, height = resolutions[opt.camRes].h}
local statPath = string.gsub(netPath,'%a+%-%d+%.net.ascii','preproc.t7.ascii')
local stat = torch.load(statPath, 'ascii')

-- Loading classes' names
local classes = {{'Torso',{1}},{'Body',{2}},{'Bckg',{3}}} -- for INRIA
local cheatClasses = {'Person', 'Background'}

-- Displaying routine
local eye = resolutions[opt.camRes].h * 3 / 4
local x1  = resolutions[opt.camRes].w / 2 - eye / 2
local y1  = resolutions[opt.camRes].h / 2 - eye / 2
local x2  = resolutions[opt.camRes].w / 2 + eye / 2
local y2  = resolutions[opt.camRes].h / 2 + eye / 2
local kSize = 21
local k   = torch.Tensor(kSize,kSize):fill(1/kSize^2)
local z   = opt.camera and eye / 128 / 4 or 1 -- zoom

-- profiling timers
local timer = torch.Timer()
local t_loop = 1

local frame, crop
function main_loop()
   local input, output, w, l, c, leg
   frame = cam:forward()
   crop = image.crop(frame, x1, y1, x2, y2)
   input = image.scale(crop,'^' .. opt.imageSide)
   w = (#input)[3]
   input = image.crop(input,w/2-opt.imageSide/2,0,w/2+opt.imageSide/2,opt.imageSide)
   for c = 1,3 do
      input[c]:add(-stat.mean[c])
      input[c]:div( stat.std [c])
   end

   -- Computing prediction
   output = net:forward(input)

   -- Computing conditional probability
   c = opt.camera and z or 0
   p = {}
   p[1] = output[1] + output[2]
   p[2] = output[3]
   predictionStr = string.format('%s (%.2f%%)', cheatClasses[1], p[1]*100)
end

-- main loop:
i = 1
while true do 
   timer:reset()
   main_loop()
   t_loop = timer:time().real

   -- transmit frame to server:
   -- curl --user admin:admin --form uploadimage=@site.jpg http://elab-gpu3.ecn.purdue.edu:8080/upload.html
   if p[1] > opt.pdt then --detection_threshold
      -- write image with person to jpeg image file on a ramdisk (to save time and save the SD card!)
      local imageLocation = "/mnt/ramdisk/person.jpg"
      image.saveJPG(imageLocation, crop)
      local serverPage = "http://elab-gpu3.ecn.purdue.edu:8080/upload.html"
      local osString = "curl --user admin:admin --form uploadimage=@" .. imageLocation .. " " .. serverPage
      os.execute(osString)
   end

   -- print logbook:
   print("Iteration: ", i, " Loop fps: ", 1/t_loop, predictionStr)
   
   collectgarbage()
   i = i+1
end

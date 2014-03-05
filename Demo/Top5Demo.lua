-- This script runs a test for the top5 routine

opt = {}
opt.temp_dir = '../Train/temp-data/'

local f = opt.temp_dir .. 'test.t7'
testData = torch.load(f)

require 'nnx'
netPath = '/home/atcold/work/indoor-EYE-results/06C/model-90.net'

net = torch.load(netPath)

opt.subsample_name = 'indoor51'
dofile('../Train/Data/indoor-classes.lua')

function label(idx)
end

require 'qtwidget'
win = qtwidget.newwindow(512,512,'New window')

require 'image'
function show(idx)
   local l = testData.labels[idx]
   local c = classes[l][1]
   local leg = l .. ': ' .. c
   local out = net:forward(testData.data[idx])
   image.display{image = testData.data[idx],
                       zoom = 4,
                       win = win}
   win.widget:setWindowTitle('True label: ' .. c)
   --gnuplot.bar(out)
   --gnuplot.title(leg)
   win:rectangle(0,0,150,110)
   win:setcolor(1,1,1,.3)
   win:fill()
   win:setfontsize(15)
   local sortOut,guess = out:sort(true)
   for i = 1,5 do
      --win.painter:show(classes[guess[i]][1]..'\n')
      --print(classes[guess[i]][1])
      win:moveto(10,20 * i)
      if guess[i] == testData.labels[idx] then
         win:setcolor('red')
      else
         win:setcolor('black')
      end
      win:show(classes[guess[i]][1])
   end
end

print('Press <Ctrl> + D for switching to the next image, <Ctrl> + C + <Return> for exiting')
i = 1
while true do
   show(i)
   i = i+1
   io.read()
end

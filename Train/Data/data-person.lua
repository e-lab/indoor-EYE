----------------------------------------------------------------------
-- This script loads the INRIA person dataset
-- training data, and pre-process it to facilitate learning.
-- E. Culurciello
-- April 2013
----------------------------------------------------------------------

require 'torch'   -- torch
require 'image'   -- to visualize the dataset
require 'nn'      -- provides a normalization operator
require 'sys'

----------------------------------------------------------------------
-- parse command line arguments
if not opt then
   print '==> processing options'
   cmd = torch.CmdLine()
   cmd:text()
   cmd:text('INRIA Person Dataset Preprocessing')
   cmd:text()
   cmd:text('Options:')
   --   cmd:option('-size', 'small', 'how many samples do we load: small | full | extra')
   cmd:option('-visualize', true, 'visualize input data and weights during training')
   cmd:text()
   opt = cmd:parse(arg or {})
end

----------------------------------------------------------------------

function ls(path) return sys.split(sys.ls(path),'\n') end -- alf ls() nice function!

local path = '/home/atcold/Work/Datasets/INRIAPerson/'
----------------------------------------------------------------------
-- load or generate new dataset:

if false and paths.filep('/home/atcold/Work/Datasets/INRIAPerson/train.t7')
   and paths.filep('/home/atcold/Work/Datasets/INRIAPerson/test.t7') then

   print '==> loading previously generated dataset:'
   trainData = torch.load('/home/atcold/Work/Datasets/INRIAPerson/train.t7')
   testData = torch.load('/home/atcold/Work/Datasets/INRIAPerson/test.t7')

   trSize = trainData.data:size(1)
   teSize = testData.data:size(1)
else

   print '==> creating a new dataset from raw files:'


   local desImaX = 46 -- desired cropped dataset image size
   local desImaY = 46

   local cropTrX = 45 -- desired offset to crop images from train set
   local cropTrY = 48
   local cropTeX = 33 -- desired offset to crop images from test set
   local cropTeY = 35
   local ivch = 3
   local labelPerson = 1 -- label for person and background:

   local trainDir = path..'96X160H96/Train/pos/'
   local trainImaNumber = #ls(trainDir)
   trSize = trainImaNumber
   local testDir = path..'70X134H96/Test/pos/'
   local testImaNumber = #ls(testDir)
   teSize = testImaNumber

   trainData = {
      data = torch.Tensor(trSize, ivch, desImaX, desImaY),
      labels = torch.Tensor(trSize),
      size = function() return trSize end
   }

   -- load person data:
   for i = 1, trainImaNumber do -- we only take every second example because the others are mirror images
      img = image.loadPNG(trainDir..ls(trainDir)[i],ivch)
      trainData.data[i] = image.crop(img, cropTrX-desImaX/2, cropTrY-desImaY/2,
      cropTrX+desImaX/2, cropTrY+desImaY/2):clone()
      trainData.labels[i] = labelPerson
   end
   -- display some examples:
   image.display{image=trainData.data[{{1,128}}], nrow=16, zoom=1, legend = 'Train Data'}


   testData = {
      data = torch.Tensor(teSize, ivch,desImaX,desImaY),
      labels = torch.Tensor(teSize),
      size = function() return teSize end
   }

   -- load person data:
   for i = 1, testImaNumber do -- we only take every second example because the others are mirror images
      img = image.loadPNG(testDir..ls(testDir)[i],ivch)
      testData.data[i] = image.crop(img, cropTeX-desImaX/2, cropTeY-desImaY/2,
      cropTeX+desImaX/2, cropTeY+desImaY/2):clone()
      testData.labels[i] = labelPerson
   end
   -- display some examples:

   --save created dataset:
   torch.save(path..'person_tr.t7',trainData)
   torch.save(path..'person_te.t7',testData)
end

-- Displaying the dataset architecture ---------------------------------------
print('Training Data:')
print(trainData.data:size())
print()

print('Test Data:')
print(testData)
print()


image.display{image=testData.data[{{1,128}}], nrow=16, zoom=1, legend = 'Test Data'}
-- Preprocessing -------------------------------------------------------------
--dofile 'preprocessing.lua'

trainData.size = function() return trSize end
testData.size = function() return teSize end

-- Exports -------------------------------------------------------------------
return {
   trainData = trainData,
   testData = testData
}

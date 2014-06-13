require 'cutorch'
require 'xlua'
require 'cunn'
require 'nnx'
require 'gnuplot'
require 'optim'

torch.setdefaulttensortype('torch.FloatTensor')
local inputSize = 2
local outputSize = 2

local model = nn.Sequential()
model:add(nn.Linear(inputSize,20))
model:add(nn.Threshold(0,0))
model:add(nn.Linear(20, outputSize))

local modelCUDA = model:clone():cuda()

local logSoft = nn.LogSoftMax()
local loss    = nn.ClassNLLCriterion()

local w, dE_dw = model:getParameters()
local wCUDA, dE_dwCUDA = modelCUDA:getParameters()

print('model weights', w:type())
print('modelCUDA weights', wCUDA:type())

-- simulate batch
local ims     = torch.FloatTensor({{0,0},{0,1},{1,0},{1,1}})
local imsCUDA = ims:clone():cuda()
-- local targets = torch.FloatTensor({1,2,2,1}) --xor
local targets = torch.FloatTensor({2,1,1,2}) --xnor

print('batch', ims)
print('target', targets)

local ce_train_error = 0
local ce_train_errorCUDA = 0

local learningRate = 0.005
local momentum = 0
local weightDecay = 0
local learningRateDecay = 0.00000001

local optimState = {
   learningRate = learningRate,
   momentum = momentum,
   weightDecay = weightDecay,
   learningRateDecay = learningRateDecay
}
local optimStateCUDA = {
   learningRate = learningRate,
   momentum = momentum,
   weightDecay = weightDecay,
   learningRateDecay = learningRateDecay
}

local E_eval_CUDA = function (att)

   dE_dwCUDA:zero()

   local outputModelGPU = modelCUDA:forward(imsCUDA)
   cutorch.synchronize()
   local outputModelCPU = outputModelGPU:float()
   local preds = logSoft:forward(outputModelCPU)
   local E = loss:forward(preds, targets)

   ce_train_errorCUDA = ce_train_errorCUDA + E

   local dE_dy = loss:backward(preds, targets)
   local gradLogSoftCPU = logSoft:backward(outputModelCPU, dE_dy)

   -- on GPU
   local gradLogSoftGPU = gradLogSoftCPU:cuda()
   modelCUDA:backward(imsCUDA, gradLogSoftGPU)
   cutorch.synchronize()

   return E, dE_dwCUDA
end

local E_eval = function (att)

   dE_dw:zero()

   local outputModelGPU = model:forward(ims)
   local outputModelCPU = outputModelGPU
   local preds = logSoft:forward(outputModelCPU)
   local E = loss:forward(preds, targets)

   ce_train_error = ce_train_error + E

   local dE_dy = loss:backward(preds, targets)
   local gradLogSoftCPU = logSoft:backward(outputModelCPU, dE_dy)

   -- on GPU
   local gradLogSoftGPU = gradLogSoftCPU
   model:backward(ims, gradLogSoftGPU)

   return E, dE_dw
end

local k = 1
local nb_iter = 10000
while ce_train_errorCUDA < 100 and k < nb_iter do

   if (k%5 == 0) then
      xlua.progress(k, nb_iter)
   end
   ce_train_error = 0
   ce_train_errorCUDA = 0

   optim.sgd(E_eval, w, optimState) 

   optim.sgd(E_eval_CUDA, wCUDA, optimStateCUDA)

   k = k + 1
end
print('Done', k, ce_train_error, ce_train_errorCUDA)

local diff = w - wCUDA:float()

gnuplot.figure(1)
gnuplot.hist(diff)
gnuplot.title('Difference Weight')

print 'Test'

local _, r = model:forward(torch.FloatTensor({0,0})):max(1)
print('0 xnor 0 =', r[1] - 1)

_, r = model:forward(torch.FloatTensor({0,1})):max(1)
print('0 xnor 1 =', r[1]- 1)

_, r = model:forward(torch.FloatTensor({1,0})):max(1)
print('1 xnor 0 =', r[1]- 1)

_, r = model:forward(torch.FloatTensor({1,1})):max(1)
print('1 xnor 1 =', r[1]- 1)

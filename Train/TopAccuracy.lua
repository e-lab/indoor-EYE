local TopAccuracy = {__index=TopAccuracy}

setmetatable(TopAccuracy, TopAccuracy)

function TopAccuracy:__call(nclasses_, ntop_)
   local self = {}
   setmetatable(self, {__index=TopAccuracy})

   self.ntop = ntop_ or 1
   self.nclasses = nclasses_
   self.mat = torch.IntTensor(self.ntop + 1):fill(0)
   self.result = torch.FloatTensor(self.ntop)

   return self
end

function TopAccuracy:add(prediction, target)
   if type(prediction) == 'number' then
      error(string.format("Can't compute top%d with only one prediction", self.ntop))
   elseif type(target) == 'number' then
      -- prediction is a vector, then target assumed to be an index
      self.prediction_1d = self.prediction_1d or torch.FloatTensor(self.nclasses)
      self.prediction_1d:copy(prediction)
      -- if we can have the 5 highest it could be better...
      local _,prediction = self.prediction_1d:sort(true)
      local p = 1
      while (p <= self.ntop and not(prediction[p] == target)) do
         p = p + 1
      end
      self.mat[p] = self.mat[p] + 1
   else
      error 'Has to be done...'
   end
end

function TopAccuracy:batchAdd(predictions, targets)
   local preds, targs
   if predictions:dim() == 1 then
      error(string.format("Can't compute top%d with only one prediction", self.ntop))
   elseif predictions:dim() == 2 then
      -- prediction is a matrix of class likelihoods
      local tmp = torch.FloatTensor(predictions:size()):copy(predictions)
      _,preds = tmp:sort(2, true)
   else
      error("predictions has invalid number of dimensions")
   end
   if targets:dim() == 1 then
      -- targets is a vector of classes
      targs = torch.FloatTensor(targets:size())
      targs:copy(targets)
   elseif targets:dim() == 2 then
      -- targets is a matrix of one-hot rows
      targs = torch.FloatTensor(targets:size(1))
      targs:copy(targets)
      targs:resize(targs:size(1))
   else
      error("targets has invalid number of dimensions")
   end

   --loop over each pair of indices
   for i = 1, preds:size(1) do
      local p = 1
      while (p <= self.ntop and not(preds[i][p] == targs[i])) do
         p = p + 1
      end
      self.mat[p] = self.mat[p] + 1
   end
end

function TopAccuracy:zero()
   self.mat:fill(0)
   self.result:fill(0)
end

function TopAccuracy:update()
   local tmp = 0
   local total = self.mat:sum()
   for t = 1, self.ntop do
      tmp = tmp + self.mat[t]
      self.result[t] = tmp/total
   end
end

return TopAccuracy

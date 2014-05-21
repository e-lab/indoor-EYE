local Threads = require 'threads'
local sdl = require 'sdl2'
local MyThreads = {__index=MyThreads}

sdl.init(0)

setmetatable(MyThreads, MyThreads)

function MyThreads:__call(nThreads, ...)
   local self = {nThreads=nThreads, threads={}, com={}}

   setmetatable(self, {__index=MyThreads})

   for thread=1, nThreads do
      self.threads[thread] = Threads(1, function () threadId = thread end, ...)
      self.com[thread] = 0
   end

   return self
end

function MyThreads:addjob(callback, endcallback, ...)
   local threadId = 1
   while (threadId <= self.nThreads and self.com[threadId] > 0) do
      threadId = threadId + 1
   end

   if (threadId > self.nThreads) then
      return 0
   end

   self.com[threadId] = 1
   self.threads[threadId]:addjob(callback, endcallback, threadId, ...)

   return threadId
end

function MyThreads:synchronize(threadId)
   if (threadId > #self.threads or threadId < 1) then
      error "This Thread doesn't exist"
   end

   self.threads[threadId]:synchronize()
   self.com[threadId] = 0
end

function MyThreads:terminate()
   for _, thread in ipairs(self.threads) do
      thread:terminate()
   end
end

return MyThreads

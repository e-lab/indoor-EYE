require 'optim'

local function recover_loggers(epochInit)
   local tmpAcc = {}
   local tmpCE = {}
   local tmpAcc5 = {}
   if (paths.filep(opt.save_dir .. 'accuracy.log')) then
      local file = io.open(opt.save_dir .. 'accuracy.log', r)
      local tmpVal = file:read("*line")  -- skip the header
      for ep = 1, epochInit do
         tmpVal = file:read("*number")  -- train
         table.insert(tmpAcc, tmpVal)
         tmpVal = file:read("*number")  -- test
         table.insert(tmpAcc, tmpVal)
      end
      io.close(file)
   end
   if (paths.filep(opt.save_dir .. 'cross-entropy.log')) then
      local file = io.open(opt.save_dir .. 'cross-entropy.log', r)
      local tmpVal = file:read("*line")  -- skip the header
      for ep = 1, epochInit do
         tmpVal = file:read("*number")  -- train
         table.insert(tmpCE, tmpVal)
         tmpVal = file:read("*number")  -- test
         table.insert(tmpCE, tmpVal)
      end
      io.close(file)
   end
   if (paths.filep(opt.save_dir .. 'accuracy-5.log')) then
      local file = io.open(opt.save_dir .. 'accuracy-5.log', r)
      local tmpVal = file:read("*line")  -- skip the header
      for ep = 1, epochInit do
         for log = 1, 2 do
            tmpVal = file:read("*number")  -- train
            table.insert(tmpAcc5, tmpVal)
            tmpVal = file:read("*number")  -- test
            table.insert(tmpAcc5, tmpVal)
         end
      end
      io.close(file)
   end

   -- (3) init loggers
   local logger = optim.Logger(opt.save_dir .. 'accuracy.log')
   local ce_logger = optim.Logger(opt.save_dir .. 'cross-entropy.log')
   local logger_5 = optim.Logger(opt.save_dir .. 'accuracy-5.log')
   logger_5:setNames({'% train top5', '% test top5',
   '% train top1', '% test top1'})

   -- (4) load backup data
   for ep = 1, epochInit do
      logger:add{['% train accuracy'] = tmpAcc[2*ep-1], ['% test accuracy'] = tmpAcc[2*ep]}
      ce_logger:add{['ce train error'] = tmpCE[2*ep-1], ['ce test error'] = tmpCE[2*ep]}
      logger_5:add{tmpAcc5[4*ep-3], tmpAcc5[4*ep-2],
      tmpAcc5[4*ep-1], tmpAcc5[4*ep]}
   end

   return logger, ce_logger, logger_5
end

-- stat file, models and training are logging informations
local statFile
--init logger for train and test accuracy
local logger
--init logger for train and test cross-entropy error
local ce_logger
--init logger for train and test accuracy-5
local logger_5

if (opt.network == 'N/A' or opt.network == 'model-0.net') then
   -- Open file in re-write mode (NOT append)
   statFile = io.open(opt.save_dir .. 'stat.txt','w+')

   logger = optim.Logger(opt.save_dir .. 'accuracy.log')
   ce_logger = optim.Logger(opt.save_dir .. 'cross-entropy.log')
   logger_5 = optim.Logger(opt.save_dir .. 'accuracy-5.log')
   logger_5:setNames({'% train top5', '% test top5',
   '% train top1', '% test top1'})

else
   statFile = io.open(opt.save_dir .. 'stat.txt','a')

   statFile:write('\n')
   statFile:write('-------------------------------------------------------------------------------\n')
   statFile:write('------------------------------------ Restart ----------------------------------\n')
   statFile:write('-------------------------------------------------------------------------------\n')

   -- (2) recover loggers
   local epochInit = tonumber(string.match(opt.network, "%d+"))
   logger, ce_logger, logger_5 = recover_loggers(epochInit)
end

logger:style{['% train accuracy'] = '-', ['% test accuracy'] = '-'}
ce_logger:style{['ce train error'] = '-', ['ce test error'] = '-'}
logger_5:style{'-', '-', '-', '-'}

function getLoggers()
   return statFile, logger, ce_logger, logger_5
end

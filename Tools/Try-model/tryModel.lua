--------------------------------------------------------------------------------
-- Routing for testing models.lua
-- Alfredo Canziani, Mar 2014
--------------------------------------------------------------------------------

-- Options ---------------------------------------------------------------------
opt = {}
opt.width          = 128
opt.cuda           = true
opt.ncolors        = 3
opt.subsample_name = 'indoor51'
opt.dropout        = 0.5
opt.inputDO        = 0.2
opt.step           = false
opt.batchSize      = 128

-- Requires --------------------------------------------------------------------
require 'nnx'
if opt.cuda then
   if step then
      io.write('Requiring cunn')
      io.read()
   end
   require 'cunn'
end
require 'eex'
data_folder = eex.datasetsPath() .. 'originalDataset/'

-- Requiring models-------------------------------------------------------------
package.path = package.path .. ';../../Train/?.lua'

if step then
   io.write('Requiring classes')
   io.read()
end
require 'Data/indoor-classes'

if step then
   io.write('Requiring model')
   io.read()
end
statFile = io.open('/tmp/temp','w')
require 'models'

model, loss, dropout, memory = get_model1()

-- Conversion function ---------------------------------------------------------
function inMB(mem)
   mem[0]=mem[0]*4/1024^2
   for a,b in pairs(mem.submodel1.val) do mem.submodel1.val[a] = b*4/1024^2 end
   for a,b in pairs(mem.submodel2.val) do mem.submodel2.val[a] = b*4/1024^2 end
end

inMB(memory)

-- Plotting memory weight ------------------------------------------------------
-- Allocating space
mem = torch.Tensor(1 + #memory.submodel1.str+1 + #memory.submodel2.str+1)
x = torch.linspace(1,mem:size(1),mem:size(1))

-- Serialise <memory> table
i=1; labels = '"OH" ' .. i -- OverHead
mem[i] = memory[0]

i=2; labels = labels .. ', "OH1" ' .. i -- OverHead <submodel1>
mem[i] = memory.submodel1.val[0]
for a,b in ipairs(memory.submodel1.str) do -- Building xtick labels
   i = i + 1
   labels = labels .. ', "' .. b .. '" ' .. i
end
mem[{ {3,3+#memory.submodel1.str-1} }] = torch.Tensor(memory.submodel1.val)

i = i + 1; labels = labels .. ', "OH2" ' .. i -- OverHead <submodel2>
mem[i] = memory.submodel2.val[0]
for a,b in ipairs(memory.submodel2.str) do -- Building xtick labels
   i = i + 1
   labels = labels .. ', "' .. b .. '" ' .. i
end
mem[{ {3+#memory.submodel1.str+1,mem:size(1)} }] = torch.Tensor(memory.submodel2.val)

print(string.format('The network training will allocate up to %.2d MB', mem:sum()))
-- Plotting
gnuplot.plot('Memory usage [MB]', x, mem, '|')
gnuplot.raw('set xtics (' .. labels .. ')')
gnuplot.axis{0,mem:size(1)+1,0,''}
gnuplot.grid(true)

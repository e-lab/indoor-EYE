-- Requires ------------------------------------------------------------------
require 'sys'
eex = {}

-- Functions -----------------------------------------------------------------

-- eex.ls - unix listing command
function eex.ls(path) return sys.split(sys.ls(path),'\n') end

-- eex.datasetPath - getting the environmental $EEX_DATASETS variable
function eex.datasetsPath()
   local ds
   ds = os.getenv("EEX_DATASETS")
   if not ds then
      io.write [[

******************************************************************************
   WARNING - $EEX_DATASETS environment variable is not defined
******************************************************************************
   Please define an environment $EEX_DATASETS variable

      $ export EEX_DATASETS=datasetsDirectoryOnCurrentMachine/

   and add it to your <.bashrc> configuration file in order to do not
   visualise this WARNING message again.
   As long as <datasetsDirectoryOnCurrentMachine> in not included by
   quotations (i.e. it is not a string), you can use shortcuts such as
   <~/>, <$HOME/>, <$PATH/>, etc..
******************************************************************************

Please, type the dataset directory path for the current machine:
]]
      ds = io.read()
      -- add `/` at the end of the path if it's missing
      if string.sub(ds,-1) ~= '/' then ds = ds .. '/' end
      io.write '\n'
   end
   return ds
end

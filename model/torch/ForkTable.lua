local ForkTable, parent = torch.class('nn.ForkTable', 'nn.Module')

function ForkTable:__init()
   parent.__init(self)
   self.modules = {}
   self.output = {}
end

function ForkTable:add(module)
   table.insert(self.modules, module)
   return self
end

function ForkTable:get(index)
   return self.modules[index]
end

function ForkTable:size()
   return #self.modules 
end

function ForkTable:updateOutput(input)
   for i=1,#self.modules do
      self.output[i] = self.modules[i]:updateOutput(input)
   end
   return self.output
end


function ForkTable:updateGradInput(input, gradOutput)
   self.gradInput:resizeAs(input)
   self.gradInput:zero()
   for i,module in ipairs(self.modules) do
      self.gradInput:add(module:updateGradInput(input, gradOutput[i]))
   end
   return self.gradInput
end

function ForkTable:accGradParameters(input, gradOutput, scale)
   scale = scale or 1
   for i,module in ipairs(self.modules) do
      module:accGradParameters(input, gradOutput[i], scale)
   end
end

function ForkTable:accUpdateGradParameters(input, gradOutput, lr)
   lr = lr or 1
   for i,module in ipairs(self.modules) do
      module:accUpdateGradParameters(input, gradOutput[i], lr)
   end
end

function ForkTable:zeroGradParameters()
   for _,module in ipairs(self.modules) do
      module:zeroGradParameters()
   end
end

function ForkTable:updateParameters(learningRate)
   for _,module in ipairs(self.modules) do
      module:updateParameters(learningRate)
   end
end

function ForkTable:training()
   for i=1,#self.modules do
      self.modules[i]:training()
   end
end

function ForkTable:evaluate()
   for i=1,#self.modules do
      self.modules[i]:evaluate()
   end
end

function ForkTable:share(mlp,...)
   for i=1,#self.modules do
      self.modules[i]:share(mlp.modules[i],...); 
   end
end

function ForkTable:parameters()
   local function tinsert(to, from)
      if type(from) == 'table' then
         for i=1,#from do
            tinsert(to,from[i])
         end
      else
         table.insert(to,from)
      end
   end
   local w = {}
   local gw = {}
   for i=1,#self.modules do
      local mw,mgw = self.modules[i]:parameters()
      if mw then
         tinsert(w,mw)
         tinsert(gw,mgw)
      end
   end
   return w,gw
end

function ForkTable:__tostring__()
   local tab = '  '
   local line = '\n'
   local next = '  |`-> '
   local ext = '  |    '
   local extlast = '       '
   local last = '   ... -> '
   local str = 'nn.ForkTable'
   str = str .. ' {' .. line .. tab .. 'input'
   for i=1,#self.modules do
      if i == self.modules then
         str = str .. line .. tab .. next .. '(' .. i .. '): ' .. tostring(self.modules[i]):gsub(line, line .. tab .. extlast)
      else
         str = str .. line .. tab .. next .. '(' .. i .. '): ' .. tostring(self.modules[i]):gsub(line, line .. tab .. ext)
      end
   end
   str = str .. line .. tab .. last .. 'output'
   str = str .. line .. '}'
   return str
end

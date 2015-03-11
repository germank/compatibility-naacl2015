local MultiCriterionTable, parent = torch.class('nn.MultiCriterionTable', 'nn.Criterion')

function MultiCriterionTable:__init()
   parent.__init(self)
   self.criterions = {}
   self.weights = torch.DoubleStorage()
   self.gradInput = {}
end

function MultiCriterionTable:add(criterion, weight)
   weight = weight or 1
   table.insert(self.criterions, criterion)
   self.weights:resize(#self.criterions, true)
   self.weights[#self.criterions] = weight
   return self
end

function MultiCriterionTable:get(i)
   return self.criterions[i]
end


local v = false
local verbose = function(msg)
    if v then print(msg) end
end
local clear = function()
    if v then os.execute('clear') end
end

local a = math.ceil(init_train_opt.batch_size/3*2) - 10
local d = 1
local s = 19
function MultiCriterionTable:updateOutput(input, target)
   clear()
   verbose(target[d][{{a,a+s}}]:t())
   verbose(input[d][{{a,a+s}}]:t())
   self.output = 0
   for i=1,#self.criterions do
      self.output = self.output + self.weights[i]*self.criterions[i]:updateOutput(input[i], target[i])
   end
   return self.output
end

function MultiCriterionTable:updateGradInput(input, target)
   for i=1,#self.criterions do
      self.gradInput[i] = self.criterions[i]:updateGradInput(input[i], target[i]):mul(self.weights[i])
   end
   verbose(self.gradInput[d][{{a,a+s}}]:t())
   return self.gradInput
end

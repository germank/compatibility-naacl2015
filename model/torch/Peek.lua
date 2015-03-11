local Peek, parent = torch.class('nn.Peek', 'nn.Module')

function Peek:__init(mod)
   parent.__init(self)
  
   self.mod = mod
   self.weight = mod.weight
   self.gradWeight = mod.gradWeight
   
   -- state
   -- self.gradInput:resize(inputSize)
   self.output = mod.output

end
 
function Peek:reset()
    self.mod:reset()
end

function Peek:updateOutput(input)
    print(input)
    return self.mod:updateOutput(input)
end

function Peek:updateGradInput(input, gradOutput)
    return self.mod(input, gradOutput)
end

function Peek:accGradParameters(input, gradOutput, scale)
    return self.mod:accGradParameters(input, gradOutput, scale)
end

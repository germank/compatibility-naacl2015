require 'torch'
require 'nn'
require 'Peek'

models.l1_mono = function(opt)
    local model = nn.Sequential()
    model:add(nn.SelectTable(1))
    model:add(nn.Linear(opt.visible_size, 1))

    return model
end


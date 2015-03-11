require 'torch'
require 'nn'
require 'Peek'

models.l1_direct = function(opt)
    local model = nn.Sequential()
    model:add(nn.JoinTable(1, 1))
    model:add(nn.Linear(opt.visible_size*2, 1))

    return model
end


require 'torch'
require 'nn'
require 'Peek'

models.l2_mono = function(opt)
    local model = nn.Sequential()
    model:add(nn.SelectTable(1))
    model:add(nn.Linear(opt.visible_size, opt.hidden_size))
    model:add(nn.Sigmoid())
    model:add(nn.Linear(opt.hidden_size, 1))

    return model
end


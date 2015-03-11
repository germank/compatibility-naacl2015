require 'torch'
require 'nn'
require 'nn2d'
require 'Peek'

models.l2_direct = function(opt)
    local model = nn.Sequential()

    local left = nn.Linear(opt.visible_size, opt.hidden_size)
    local right = left:clone('weight', 'bias', 'gradWeight', 'gradBias')

    --Map the features with weight sharing
    local mapping = nn.ParallelTable()
    mapping:add(left)
    mapping:add(right)
    model:add(mapping)
    local sigmoid = nn.ParallelTable()
    sigmoid:add(nn.Sigmoid())
    sigmoid:add(nn.Sigmoid())

    model:add(sigmoid)

    --Linear mapping to produce a score
    model:add(nn.JoinTable(1, 1))
    model:add(nn.Linear(2*opt.hidden_size, 1))

    return model
end



require 'torch'
require 'nn'
require 'nn2d'
require 'models.Interaction'
require 'Peek'


models.l2_interaction = function(opt)
    local model = nn.Sequential()

    local left = nn.Linear(opt.visible_size, opt.hidden_size)
    local right = left:clone('weight', 'bias', 'gradWeight', 'gradBias')

    local mapping = nn.ParallelTable()
    mapping:add(left)
    mapping:add(right)
    model:add(mapping)
    local sigmoid = nn.ParallelTable()
    sigmoid:add(nn.Sigmoid())
    sigmoid:add(nn.Sigmoid())

    model:add(sigmoid)

    model:add(models.InteractionModule(opt.hidden_size))
    model:add(nn.Linear(opt.hidden_size, 1))

    return model
end


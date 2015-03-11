require 'torch'
require 'nn'
require 'nn2d'
require 'Peek'

models.l2_interaction_direct = function(opt)
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

    --Interaction layer
    local layer2 = nn.ConcatTable()
    layer2:add(models.InteractionModule(opt.hidden_size))
    --pass on the mapped vectors
    layer2:add(nn.Identity())
    model:add(layer2)

    --Linear mapping to produce a score
    model:add(nn.FlattenTable())
    model:add(nn.JoinTable(1, 1))
    model:add(nn.Linear(3*opt.hidden_size, 1))

    return model
end


require 'torch'
require 'nn'
require 'nn2d'
require 'models.Interaction'
require 'Peek'


models.l1_interaction_direct = function(opt)

    local model = nn.Sequential()
    local layer1 = nn.ConcatTable()
    layer1:add(models.InteractionModule(opt.visible_size))
    --pass on the mapped vectors
    layer1:add(nn.Identity())
    model:add(layer1)

    --Linear mapping to produce a score
    model:add(nn.FlattenTable())
    model:add(nn.JoinTable(1, 1))
    model:add(nn.Linear(3*opt.visible_size, 1))

    return model
end



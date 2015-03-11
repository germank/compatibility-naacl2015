require 'nn'
require 'nn2d'

function models.InteractionModule(hidden_size)
    local mod = nn.Sequential()
    local feature_weighting = nn.ParallelTable()
    feature_weighting:add(nn.CMul2D(hidden_size))
    feature_weighting:add(nn.CMul2D(hidden_size))
    mod:add(feature_weighting)
    mod:add(nn.CAddTable())
    mod:add(nn.Add(hidden_size))
    mod:add(nn.Sigmoid())
    return mod
end

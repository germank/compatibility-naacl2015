require 'torch'
require 'nn'
require 'Peek'

models.cosine_wrapper = function(model)
    local wrap = nn.Sequential()
    wrap:add(nn.ParallelTable())
    --compute the model score
    wrap:get(1):add(model)
    --pass through the cosine
    wrap:get(1):add(nn.Identity())
    --join them
    wrap:add(nn.JoinTable(1, 1))
    wrap:add(nn.Linear(2, 1))

    return wrap
end


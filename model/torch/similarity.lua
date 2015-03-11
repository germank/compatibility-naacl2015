require 'torch'
require 'models'
require 'nn'
require 'nn2d.CosineDistance'


models.similarity = function(opt)
    local model = nn.Sequential()
    --compute cosine distance
    model:add(nn.CosineDistance2D())
    --add bias term
    model:add(nn.Add(1))
    --classify
    model:add(nn.Sigmoid())

    return model
end

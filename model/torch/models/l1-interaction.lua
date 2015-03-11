require 'torch'
require 'nn'
require 'nn2d'
require 'models.Interaction'
require 'Peek'


models.l1_interaction = function(opt)
    local model = nn.Sequential()
    model:add(models.InteractionModule(opt.visible_size))
    model:add(nn.Linear(opt.visible_size, 1))

    return model
end



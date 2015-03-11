require 'optim'
require 'nn'
--require 'ForkTable'
require 'MultiCriterionTable'


function ds_add_classes(ds, opt) 
    local input, rs = unpack(ds)

    local negative, positive = torch.Tensor(rs:size()), torch.Tensor(rs:size())

    negative:fill(negative_class_lbl)
    positive:fill(negative_class_lbl)

    for i=1,rs:size(1) do
        if rs[{i,1}] <= opt.class_min_thr then
            negative[{i,1}] = positive_class_lbl
        end
        if rs[{i,1}] > opt.class_max_thr then
            positive[{i,1}] = positive_class_lbl
        end
    end

    return {input, rs, positive, negative}
end

function emphasizing_order(ds_size, batch_size, positive, negative)
    local n, o, p = {}, {}, {}
    for i=1,positive:size(1) do
        if positive[{i,1}] == 1 then
            table.insert(p, i)
        end
        if negative[{i,1}] == 1 then
            table.insert(n, i)
        end
        if negative[{i,1}] == 0 and positive[{i,1}] == 0 then
            table.insert(o, i)
        end
    end
    local res ={}
    for t=1,ds_size,batch_size do
        if ds_size - t < batch_size then break end
        for i,x in pairs {n,o,p} do
            fill_samples(res, t + math.floor((i-1)*(batch_size/3)), math.ceil(batch_size/3), x)
        end
    end
    return res
end

function fill_samples(res, s, chunk_size, x)
    local r = torch.randperm(#x)
    for i=1,chunk_size do
        res[s+i-1] = x[r[i]]
    end
end

function train(model, ds, init_train_opt, train_opt, opt)
    
    local inputs, rs, positive, negative = unpack(ds_add_classes(ds, opt))

    local ds_size = rs:size(1)


    local targets, criterion, ext_model = extend_model_head(model, {rs, positive, negative})

    local optim_state, optim_func = get_training_method(init_train_opt)
    local batch_size = init_train_opt.batch_size or targets[1]:size(1) / init_train_opt.nbatchs
    for i=1,init_train_opt.max_epochs do
        local perm
        if opt.emphasizing_scheme then
            perm = emphasizing_order(ds_size, batch_size, positive, negative)
        else
            perm = torch.randperm(ds_size)
        end
        for t=1,ds_size,batch_size do
            if ds_size - t + 1 < batch_size then break end
            local batch_func = opt.use_cosine and get_batch_cosine or get_batch
            local batch = batch_func(t, inputs, targets, perm, batch_size, ds_size)

            do_train_batch(ext_model, criterion, batch, optim_func, optim_state)
        end
    end

    local optim_state, optim_func = get_training_method(train_opt)
    local x, dl_dx = ext_model:getParameters()
    local old_x = torch.Tensor(x:size())
    for i=1,train_opt.max_epochs do 
        old_x:copy(x)
        do_train_batch(ext_model, criterion, {inputs, targets}, optim_func, optim_state)
        x, dl_dx = ext_model:getParameters()
        local step_size = (old_x - x):norm()
        print('Step size: '..step_size)
        if step_size < train_opt.epsilon then
            print('Error change below tolerance: '..(old_x- x):norm())
            break
        end
    end

    return ext_model
end

function extend_model_head(model, ds_targets)
    local rs, positive, negative = unpack(ds_targets)

    local targets = {}
    if obj['corr'] then
        table.insert(targets, rs)
    end
    if obj['pos'] then
        table.insert(targets, positive)
    end
    if obj['neg'] then
        table.insert(targets, negative)
    end

    --combination of criteria
    local criterion = nn.MultiCriterionTable()
    if obj['corr'] then
        criterion:add(nn.MSECriterion(), 1)
    end

    local npos = positive:sum()
    local nneg = negative:sum()
    if obj['pos'] then
        local wpos = (npos + nneg) / (2*npos)
        criterion:add(nn.BCECriterion(), opt.coef_class_obj/2)
    end
    if obj['neg'] then
        local wneg = (npos + nneg) / (2*nneg)
        criterion:add(nn.BCECriterion(), opt.coef_class_obj/2)
    end

    local ext_model = nn.Sequential()
    --compute the scores that will be passed to the multiple criteria
    local mod_criterion_scores = nn.ConcatTable()
    if obj['corr'] then
        mod_criterion_scores:add(nn.Identity())
    end
    local pos_score = nn.Sequential()
    if obj['pos'] then
        pos_score:add(nn.Linear(1,1))
        pos_score:add(nn.Sigmoid())
        mod_criterion_scores:add(pos_score)
    end
    local neg_score = nn.Sequential()
    if obj['neg'] then
        neg_score:add(nn.Linear(1,1))
        neg_score:add(nn.Sigmoid())
        mod_criterion_scores:add(neg_score)
    end
    ext_model:add(model)
    ext_model:add(mod_criterion_scores)
    return targets, criterion, ext_model 
end

function get_training_method(train_opt) 
    local optim_state, optim_func
    if train_opt.algorithm ==  'sgd' then
        optim_state = {
           learningRate = train_opt.learning_rate,
           learningRateDecay = train_opt.learning_rate_decay,
           weightDecay = 0,
           momentum = 0,
           verbose = train_opt.verbose
           }
        optim_func = optim.sgd
    elseif train_opt.algorithm == 'lbfgs' then
        optim_state = {
            maxIter = train_opt.max_iter,
            learningRate = train_opt.learning_rate,
            verbose = train_opt.verbose
        }
        optim_func = optim.lbfgs
    elseif train_opt.algorithm == 'cg' then
        optim_state = {
            maxIter = train_opt.max_iter,
            verbose = train_opt.verbose
        }
        optim_func = optim.cg
    elseif train_opt.algorithm == 'adagrad' then
        optim_state = {
            learningRate=train_opt.learning_rate,
            verbose=train_opt.verbose
        }

        optim_func = optim.adagrad
    else
        error('Unknown algorithm: ', train_opt.algorithm)
    end
    return optim_state, optim_func
end


function get_batch(t, inputs, targets, perm, batch_size, ds_size)
    local batch_inputs = {}
    local batch_targets = {}
    for k=1,#inputs do batch_inputs[k]=torch.Tensor(batch_size, inputs[1]:size(2)) end
    for k=1,#targets do batch_targets[k]=torch.Tensor(batch_size, targets[1]:size(2)) end

    for i = t,math.min(t+batch_size-1,ds_size) do
        -- load new sample
        for k=1,#inputs do
            local k_input = inputs[k][perm[i]]
            batch_inputs[k][i-t+1]:copy(k_input)
        end
        for k=1,#targets do
            local k_target = targets[k][perm[i]]
            batch_targets[k][i-t+1]:copy(k_target)
        end
    end

    return {batch_inputs, batch_targets}
end

--FIXME: make a recursive version of the previous function to handle this 
--case
function get_batch_cosine(t, inputs, targets, perm, batch_size, ds_size)
    local batch_inputs = {{}}
    local batch_targets = {}
    for k=1,#inputs[1] do batch_inputs[1][k]=torch.Tensor(batch_size, inputs[1][k]:size(2)) end
    batch_inputs[2]=torch.Tensor(batch_size, inputs[2]:size(2))
    for k=1,#targets do batch_targets[k]=torch.Tensor(batch_size, targets[1]:size(2)) end

    for i = t,math.min(t+batch_size-1,ds_size) do
        -- load new sample
        for k=1,#inputs[1] do
            local k_input = inputs[1][k][perm[i]]
            batch_inputs[1][k][i-t+1]:copy(k_input)
        end
        local k_input = inputs[2][perm[i]]
        batch_inputs[2][i-t+1]:copy(k_input)
        for k=1,#targets do
            local k_target = targets[k][perm[i]]
            batch_targets[k][i-t+1]:copy(k_target)
        end
    end

    return {batch_inputs, batch_targets}
end

function do_train_batch(model, criterion, batch, optim_func, optim_state)
    local inputs, targets = unpack(batch)

    local x, dl_dx = model:getParameters()


    local feval = function(x_new)
        -- set x to x_new, if different
        -- (in this simple example, x_new will typically always point to x,
        -- so this copy is never happening)
        if x ~= x_new then
            x:copy(x_new)
        end

        --reset the gradients (!!!)
        dl_dx:zero()

        -- evaluate the loss function and its derivative wrt x, for that sample
        local criterion_scores = model:forward(inputs)

        
        local loss_x = criterion:forward(criterion_scores, targets)
        --local r = stats.pearson(criterion_scores[1], targets)
        --print("Correlation: "..r)
        io.stdout:write('Loss: '..loss_x.."\n" )
        model:backward(inputs, criterion:backward(criterion_scores, targets))
        --Add L2 Regularization
        if opt.coef_L2 ~= 0  then
            loss_x = loss_x + opt.coef_L2 * torch.norm(x, 2)^2 / 2
            dl_dx:add(x:clone():mul(opt.coef_L2))
        end
     
        -- return loss(x) and dloss/dx
        return loss_x, dl_dx
    end

    local optim_x, ls = optim_func(feval, x, optim_state)
end

function predict(model, test_input)
    return model:forward(test_input)
end


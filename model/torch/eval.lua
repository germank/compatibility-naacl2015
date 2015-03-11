require 'stats'
require 'train'


function cv_eval(model, dss, init_train_opt, train_opt, opt)
    local scores = {}
    for i, ds in pairs(dss) do
        local score = eval(model, ds, init_train_opt, train_opt, opt)
        table.insert(scores, score)
    end
    return scores
end

function eval(model, ds, init_train_opt, train_opt, opt) 
    local train_ds, test_ds = unpack(ds)
    print ('Using train ds of ' .. train_ds.length .. ' elements')
    print ('Using test ds of ' .. test_ds.length .. ' elements')

    train(model, train_ds, init_train_opt, train_opt, opt)

    local input, rs = unpack(test_ds)

    local pred_rs = predict(model, input)
    print 'Testing'
    print (torch.cat(pred_rs, rs, 2))


    local r = stats.pearson(rs, pred_rs)    

    print 'Training'
    local input, rs = unpack(train_ds)

    local pred_rs = predict(model, input)
    --print (torch.cat(pred_rs, rs, 2))
    print ('Correlation: '.. r)
    return r
    
end

function dump(t, filename)
    local f = assert(io.open(filename, 'w'))
    for i=1,t:size(1) do
        f:write(t[{i,1}]..'\n')
    end
    f:close()
end


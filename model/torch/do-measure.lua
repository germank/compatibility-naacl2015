#!/usr/bin/env luajit
require 'pl.strict'
require 'config'
require 'datasets'
require 'models'
require 'eval'




function basename(str)
    local name = string.gsub(str, "(.*/)(.*)", "%2")
    return name
end



for x=1,#arg do
    local status, errp = pcall(function() 
    local conf_file = arg[x]
    io.stderr:write('Using run configuration: '..conf_file..'\n')
    load_run_config(conf_file)
    local datasets =  get_dataset_files(opt)

    io.stderr:write('loading model from: '..model_filename..'\n')
    local model = torch.load(model_storage_dir..model_filename)
    for ds_type,ds_group in pairs(datasets) do
        for _i,ds_filename in pairs(ds_group) do
            print('Loading dataset'..ds_filename)
            local test_ds = load_dataset_vectors(ds_filename, opt, nil, true)

            local inputs = unpack(test_ds)

            local pred_rs = predict(model, inputs)

            local output_dir = test_output_dir..ds_type..'/'..model_filename.."/"
            os.execute("mkdir -p ".. output_dir)
            local output_file = output_dir .. basename(ds_filename)

            local f = assert(io.open(output_file, 'w'))
            for i=1,pred_rs:size(1) do
                local row_i = {}
                --if ds_type == 'compatibility' then
                --    table.insert(row_i, test_ds.labels[i][1])
                --    table.insert(row_i, test_ds.labels[i][2])
                --    table.insert(row_i, test_ds.labels[i][3])
                --    table.insert(row_i, test_ds.labels[i][3] <= 1.6 and "1" or "0")
                --    table.insert(row_i, test_ds.labels[i][3] > 3.7 and "1" or "0")
                --    table.insert(row_i, pred_rs[{i,1}])
                --else 
                    table.insert(row_i, test_ds.labels[i][1])
                    table.insert(row_i, test_ds.labels[i][2])
                    table.insert(row_i, pred_rs[{i,1}])
                --end

                f:write(table.concat(row_i, '\t').."\n")
            end
            f:close()
        end
    end
    end)
    if not status then print (errp) end
end

--print 'Testing'
--print (torch.cat(pred_rs, rs, 2))
--
--
--dump(rs, 'rs.txt')
--dump(pred_rs, 'pred_rs.txt')
--local r = stats.pearson(rs, pred_rs)    
--
--print 'Training'
--local w1s, w2s, rs = unpack(train_ds)
--
--local pred_rs = predict(model, {w1s, w2s})
----print (torch.cat(pred_rs, rs, 2))
--print ('Correlation: '.. r)

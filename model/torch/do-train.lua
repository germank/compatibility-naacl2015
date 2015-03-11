#!/usr/bin/env th
require 'pl.strict'
require 'config'
require 'datasets'
require 'train'
require 'models'
require 'similarity'
require 'eval'

--[[local dss = {}
for i=1,10 do
    dss[#dss] = {
        base_path .. 'dataset/final/10-fold/core.train-'..i..'.txt',
        base_path .. 'dataset/final/10-fold/core.test-'..i..'.txt'
    }
end ]]--

init_time = os.time()
if #arg>=1 then
    local conf_file = arg[1]
    load_run_config(conf_file)
end
print("Model: "..model_name)
print("Objective: "..obj_name)

local ds_dir
if opt.use_cosine then
    ds_dir = base_path .. 'model/baseline/output/compatibility-disjoint/cosine/'
else
    ds_dir = base_path .. 'dataset/final/disjoint/'
end
local ds = { 
    ds_dir .. 'train_dup.txt',
    ds_dir .. 'dev_dup.txt'
}

local ds = get_dataset_files(opt)[compatibility_ds]
ds = {ds['train'], ds['dev']}

--local dss_vectors = dataset_vector_loader(dss, opt)
local ds_vectors = dataset_vector_loader(ds, opt)

local model = models[model_name](opt)

if opt.use_cosine then
    model = models.cosine_wrapper(model)
end

local status, errp = xpcall(function()
    local scores = eval(model, ds_vectors, init_train_opt, train_opt, opt)
end, debug.traceback)

if not status then
    print(errp)
end


end_time = os.time()


print("Running time: " .. (end_time-init_time)/60 .. "min")
print("Saving model to ".. model_filename)
torch.save(model_storage_dir..model_filename, model)


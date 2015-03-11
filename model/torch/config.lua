require("my_config")
opt = {
    visible_size=400,
    hidden_size=100,
    words_file = base_path .. 'space/vocab.txt',
    vectors_file = base_path .. 'space/EN-wform.w.5.cbow.neg10.400.subsmpl_small.th7',
    class_min_thr = 1.6,
    class_max_thr = 3.7,
    coef_class_obj = 1.0,
    coef_L2 = 1e-3,
    emphasizing_scheme = true,
    use_cosine = false

}

init_train_opt = {
    max_epochs = 120,
    batch_size = 150,
    learning_rate = 5e-2,
    learning_rate_decay = 0,
    algorithm = 'adagrad',
    verbose=false
}

train_opt = {
    max_epochs = 0,
    learning_rate = 1e-3,
--    learning_rate_decay = 1e-8,
    max_iter=20,
    algorithm = 'lbfgs',
    epsilon=1e-8,
    verbose=false
}


--goal
obj = {}
obj['corr'] = false
obj['pos'] = false
obj['neg'] = false


--model
model_name = nil

positive_class_lbl = 1
negative_class_lbl = 0


obj_name = nil
model_filename = nil
model_storage_dir = "trained_models/"

test_output_dir = "test_output/"


compatibility_ds = 'compatibility-disjoint'
function get_dataset_files(opt)
    local datasets = {}
    local ds_dir
    --if opt.use_cosine then
    --    ds_dir = base_path .. 'model/baseline/output/'..compatibility_ds..'/cosine/'
    --else
        ds_dir = base_path .. 'dataset/'..compatibility_ds..'/'
    --end
    datasets[compatibility_ds] = {
        dev = ds_dir .. 'dev_dup.txt',
        train = ds_dir .. 'train_dup.txt',
        test = ds_dir .. 'test_dup.txt'
    }


    return datasets
end

--auxiliary stuff
function load_run_config(filename)
    dofile(filename)
    local obj_names = {}
    for k,t in pairs(obj) do
        if t then  table.insert(obj_names, k) end
    end
    obj_name = table.concat(obj_names, "_")
    local cosine_name = opt.use_cosine and "cos" or "nocos"
    model_filename = model_name.."-"..cosine_name.."-"..obj_name.."-"..string.gsub(string.format("%1.e",opt.coef_L2),"-","_")..'.th7'
end

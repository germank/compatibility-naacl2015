#!/usr/bin/env luajit
require 'pl.strict'
require 'config'
require 'datasets'
require 'models'
require 'eval'


local dump_output_dir = 'mapping_dump'

function basename(str)
    local name = string.gsub(str, "(.*/)(.*)", "%2")
    return name
end

local vocab = {}
vf = assert(io.open(arg[1]))
for v in vf:lines() do
    table.insert(vocab, v)
end

local revindex, vectors = load_vectors(opt)
vocab_vectors = words_to_vectors(vocab, revindex, vectors)

for x=2,#arg do
    local status, errp = pcall(function() 
    local conf_file = arg[x]
    io.stderr:write('Using run configuration: '..conf_file..'\n')
    load_run_config(conf_file)

    io.stderr:write('loading model from: '..model_filename..'\n')
    local model = torch.load(model_storage_dir..model_filename)

    local output_dir = dump_output_dir
    os.execute("mkdir -p ".. output_dir)
    local output_file = output_dir ..'/input_'.. basename(model_filename)

    local f = assert(io.open(output_file, 'w'))
    for i=1,vocab_vectors:size(1) do
        --local mapped = model:get(1):get(1):forward(vocab_vectors[i])
        local mapped = vocab_vectors[i]
        local row={}
        table.insert(row, vocab[i])
        for j=1,mapped:size(1) do
            table.insert(row, mapped[j])
        end
        f:write(table.concat(row, '\t').."\n")
    end
    f:close()
    end)
    if not status then print (errp) end
end


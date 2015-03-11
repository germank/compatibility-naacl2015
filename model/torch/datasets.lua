require 'torch'
require 'io'
require 'string'
require 'config'

local word_matcher = '([%w.\'&-/]+)'
function pairs_line_matcher(line)
    local h,c = string.match(line, word_matcher..'\t'..word_matcher)
    return {h,c}
end
function cosine_pairs_line_matcher(line)
    local w1,w2,c = string.match(line, word_matcher..'\t'..word_matcher..'\t(-?%d*%d.%d+)')
    return {w1,w2,c}
end
function ratings_line_matcher(line)
    local h,c,r = string.match(line, '(%w+)\t(%w+)\t(%d*%d.%d+)')
    return {h,c,tonumber(r)}
end
function cosine_line_matcher(line)
    local w1,w2,r,c = string.match(line, '(%w+)\t(%w+)\t(%d*%d.?%d+)\t%d\t%d\t(-?%d.%d+)')
    return {w1,w2,tonumber(r),tonumber(c)}
end

function line_validator(line, parsed_line, revindex, for_test)
    local h, c, r= unpack(parsed_line)
    if not revindex or ( revindex[h] ~= nil and revindex[c] ~= nil and (for_test or r ~= nil)) then
        return true
    elseif revindex then
        if not h or not c then
            io.stderr:write('Failed to parse line: "'.. line .. '"\n')
            print(h,c)
        else 
            if revindex[h] == nil then
                io.stderr:write('Missing word: '..h..'\n')
            end
            if revindex[c] == nil then
                io.stderr:write('Missing word: '..c..'\n')
            end
        end
    end
    return false
end

function read_dataset(filename, line_matcher, revindex, max, for_test)
    local ds = {}
    local file = io.open(filename)
    if not file then
        error('Couldn\'t find file: '..filename)
    end
    for line in file:lines() do
        local parsed_line = line_matcher(line)
        if line_validator(line, parsed_line, revindex, for_test) then
            ds[#ds+1] = parsed_line
            if max ~= nil then
                max = max -1
                if max <= 0 then break end
            end
        end
    end
    return ds
end
function load_wordlist(filename, revindex)
    local ds = {}
    local file = io.open(filename)
    local max = nil
    for line in file:lines() do
        local w = string.match(line, '(%w+)-n')
        if revindex[w] ~= nil then
            ds[#ds+1] = w
            if max ~= nil then
                max = max -1
                if max <= 0 then break end
            end
        else
            if revindex[w] == nil then
                io.stderr:write('Missing word: '..w..'\n')
            end
        end
    end
    return ds
end
function load_vectors(opt) 
    local revindex = {}
    local file = io.open(opt.words_file)
    local i=1
    for line in file:lines() do
        revindex[line] = i
        i = i + 1
    end
    local vectors = torch.load(opt.vectors_file)
    return revindex,vectors
end
function wordpairs_to_vectors(wordpairs, revindex, vectors)
    local w1s = torch.Tensor(#wordpairs, vectors:size()[2])
    local w2s = torch.Tensor(#wordpairs, vectors:size()[2])
    for i,p in ipairs(wordpairs) do
        w1s[i] = vectors[revindex[p[1]]]
        w2s[i] = vectors[revindex[p[2]]]
    end
    return w1s, w2s
end

function words_to_vectors(words, revindex, vectors)
    local vs = torch.Tensor(#words, vectors:size()[2])
    for i,w in ipairs(words) do
        if revindex[w] == nil then
            error('Word '..w..' is not in the vocabulary')
        end
        vs[i] = vectors[revindex[w]]
    end
    return vs
end

function rating_vectors(dataset, maxN)
    local n = maxN or #dataset
    local rs = torch.Tensor(n, 1)

    for i,p in ipairs(dataset) do
        rs[{i,1}] = p[3]
    end
    return rs
end

function cosine_vectors(dataset, maxN,for_test)
    local n = maxN or #dataset
    local cs = torch.Tensor(n, 1)

    for i,p in ipairs(dataset) do
        cs[{i,1}] = p[for_test and 3 or 4]
    end
    return cs
end

function load_dataset_vectors(filename, opt, maxN, for_test)
    local revindex, vectors = load_vectors(opt)

    local line_matcher
    if for_test then
        if opt.use_cosine then
            line_matcher = cosine_pairs_line_matcher
        else
            line_matcher = pairs_line_matcher
        end 
    else
        if opt.use_cosine then
            line_matcher = cosine_line_matcher
        else
            line_matcher = ratings_line_matcher
        end
    end
    local dataset = read_dataset(filename, line_matcher, revindex, maxN, for_test)
    local w1s, w2s = wordpairs_to_vectors(dataset, revindex, vectors)
    local input
    if opt.use_cosine then
        local cs = cosine_vectors(dataset, maxN,for_test) -- torch.Tensor(positive_hs:size()[1], 1):fill(1)
        input = {{w1s,w2s},cs}
    else
        input = {w1s,w2s}
    end

    local ret = {length = w1s:size()[1],labels=dataset,  input}
    if not for_test then
        local rs = rating_vectors(dataset, maxN) -- torch.Tensor(positive_hs:size()[1], 1):fill(1)
        ret[2] = rs
    end
    return ret


end

function dataset_vector_loader(dss, opt, maxN)
    if type(dss) == 'table' then
        local ret = {}
        for k, v in pairs(dss) do
            ret[k] = dataset_vector_loader(v, opt, maxN)
        end
        return ret
    else
        return load_dataset_vectors(dss, opt, maxN)
    end
end

function ds_to_libsvm(hs, cs, y) 
    for i=1,y:size()[1] do
        io.write(y[{i,1}])
        io.write(" ")
        for j=1,hs:size()[2] do
            io.write(j)
            io.write(':')
            io.write(hs[{i,j}])
            io.write(' ')
        end

        for j=1,cs:size()[2] do
            io.write(hs:size()[2] + j)
            io.write(':')
            io.write(cs[{i,j}])
            io.write(' ')
        end
        io.write('\n')
    end
end
function ds_to_arff(hs, cs, y) 
    m = y:size()[1]
    n = hs:size()[2]
    io.write('@RELATION entailment\n')

    for j=1,2*n do
        io.write('@ATTRIBUTE att_'..j..' NUMERIC\n')
    end
    io.write('@ATTRIBUTE class {-1, 1}\n')

    io.write('@DATA\n')
    for i=1,m do
        for j=1,n do
            io.write(hs[{i,j}])
            io.write(',')
        end

        for j=1,n do
            io.write(cs[{i,j}])
            io.write(',')
        end
        io.write(y[{i,1}])
        io.write('\n')
    end
end

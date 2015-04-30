stats = {}

function stats.get_sim(v1, v2)
    if v1:nDimension() == 1 then
        v1 = v1:reshape(1, v1:size()[1])
    end
    if v2:nDimension() == 1 then
        v2 = v2:reshape(1, v2:size()[1])
    end
    if v1:size()[1] % v2:size()[1] == 0 then
        v2 = v2:expandAs(v1)
    elseif v2:size()[1] % v1:size()[1] == 0 then
        v1 = v1:expandAs(v2)
    end

    return torch.cdiv(torch.cmul(v1,v2):sum(2),(torch.cmul(torch.norm(v1,2,2),torch.norm(v2,2,2))))
end

function stats.normalize(v)
    return (v - torch.mean(v))/torch.std(v)
end

function stats.scale(v, a, b)
    a = a or 0
    b = b or 1
    local xmin = torch.min(v)
    local xmax = torch.max(v)
    return (v - xmin) * (b - a)/ (xmax - xmin) + a
end

function stats.pearson(v1, v2)
    local c1 = v1 - torch.mean(v1)
    local c2 = v2 - torch.mean(v2)
    c1 = c1:view(-1)
    c2 = c2:view(-1)
    return stats.get_sim(c1, c2)[{1,1}]
end


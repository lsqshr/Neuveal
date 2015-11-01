require 'torch'
require 'dp'

-- The dataset class for learning the embedding image block and synthesised block
local volume3d, DataSource = torch.class("volume3d", "dp.DataSource")

function volume3d:__init(opt)
    -- if opt.load_all then
    --     self:loadall(opt)
    -- end
        
    DataSource.__init(self, {
        train_set=self:trainSet(), valid_set=self:validSet(),
        test_set=self:testSet(), input_preprocess=input_preprocess,
        target_preprocess=target_preprocess})
end


function volume3d:loadbatch(opt, b)
    -- load
    local data = {}
    data.inputs = {}
    data.targets = {}
    -- local imgdata = {}
    -- local syndata = {}

    -- Load Data and Make the dp.BaseSet
    local mattorch = require 'mattorch'
    startidx = 1 + (b - 1) * opt.batchsize
    endidx = startidx + opt.batchsize - 1
    print(string.format('loading blocks from %s/blocks%d-%d.mat', opt.datapath, startidx, endidx))
    data.inputs = mattorch.load(string.format('%s/blocks%d-%d.mat', opt.datapath, startidx, endidx))
    data.inputs = data.inputs['blocks']
    maxinput = torch.max(data.inputs)
    maxmat = torch.Tensor(data.inputs:size(1), data.inputs:size(2),
                          data.inputs:size(3), data.inputs:size(4)):fill(1/maxinput)
    data.inputs = data.inputs:cmul(maxmat) -- Normalise the inputs
    dsz = #data.inputs
    data.inputs:resize(dsz[1], 1, dsz[2], dsz[3], dsz[4])

    data.targets = mattorch.load(string.format('%s/gt%d-%d.mat', opt.datapath, startidx, endidx))
    data.targets = data.targets['gt']

    data.coord = mattorch.load(string.format('%s/coord%d-%d.mat', opt.datapath, startidx, endidx))
    data.coord = data.coord['coord']
    -- data.inputs:resize(dsz[1], 1, dsz[2], dsz[3], dsz[4])

    assert((#data.inputs)[1] == (#data.targets)[1] and (#data.targets)[1] == (#data.coord)[1])

    return data
end

return loader
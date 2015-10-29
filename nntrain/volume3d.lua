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
    -- print(string.format('loading blocks from %s/blocks%d-%d.mat', opt.datapath, startidx, endidx))
    data.inputs = mattorch.load(string.format('%s/blocks%d-%d.mat', opt.datapath, startidx, endidx))
    data.inputs = data.inputs['blocks']
    dsz = #data.inputs
    -- print('data.inputs size:')
    -- print(dsz)
    -- print(string.format('loading gt from %s/gt%d-%d.mat', opt.datapath, startidx, endidx))
    data.targets = mattorch.load(string.format('%s/gt%d-%d.mat', opt.datapath, startidx, endidx))
    data.targets = data.targets['gt']
    data.inputs:resize(dsz[1], 1, dsz[2], dsz[3], dsz[4])

    assert((#data.inputs)[1] == (#data.targets)[1])

    -- padinputs = torch.Tensor(data.imgdata:size(1), 1, opt.kernelSize1[1] - 1 + data.imgdata:size(3),
    --                              opt.kernelSize1[2] - 1 + data.imgdata:size(4),
    --                              opt.kernelSize1[3] - 1 + data.imgdata:size(5)):fill(0)
    -- padinputs[{{}, {1}, {(opt.kernelSize1[1] - 1) / 2 + 1, (opt.kernelSize1[1] - 1) / 2 + data.imgdata:size(3)},
    --               {(opt.kernelSize1[2] - 1) / 2 + 1, (opt.kernelSize1[2] - 1) / 2 + data.imgdata:size(4)},
    --               {(opt.kernelSize1[3] - 1) / 2 + 1, (opt.kernelSize1[3] - 1) / 2 + data.imgdata:size(5)}}] = data.imgdata
    -- print('input size', #data.inputs)
    -- print('padinput size', #padinputs)
    -- print('target size', #data.targets)

    return data
end

-- function volume3d:createDataSet(inputs, targets, which_set)
--     local input_v, target_v = dp.DataView(), dp.DataView()
--     input_v:forward('bchwd', inputs)
--     target_v:forward('bchwd', targets)
--     local ds = dp.DataSet{inputs=input_v, targets=target_v, which_set=which_set}
--     ds:ioShapes('bchwd', 'bchwd')
--     return ds
-- end

return loader
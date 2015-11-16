require 'torch'
require 'dp'

-- The dataset class for learning the embedding image block and synthesised block
local volume3d, DataSource = torch.class("volume3d", "dp.DataSource")

function volume3d:__init(opt, blocklist, gtlist, coordlist)
    --The filelist contains a table of batch filenames
    self.blocklist = blocklist
    self.gtlist = gtlist
    self.coordlist = coordlist

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

    -- Load Data and Make the dp.BaseSet
    -- local mattorch = require 'mattorch'

    data.inputs = torch.load(self.blocklist[b])
    -- data.inputs = data.inputs['blocks']
    maxinput = torch.max(data.inputs)
    maxmat = torch.Tensor(data.inputs:size(1), data.inputs:size(2),
                          data.inputs:size(3), data.inputs:size(4)):fill(1/maxinput)
    data.inputs = data.inputs:cmul(maxmat) -- Normalise the inputs
    dsz = #data.inputs
    data.inputs:resize(dsz[1], 1, dsz[2], dsz[3], dsz[4])

    data.targets = torch.load(self.gtlist[b])
    -- data.targets = data.targets['gt']

    data.coord = torch.load(self.coordlist[b])
    -- data.coord = data.coord['coord']

    assert((#data.inputs)[1] == (#data.targets)[1] and (#data.targets)[1] == (#data.coord)[1])

    return data
end

function volume3d:get_nbatch()
    return #self.blocklist
end

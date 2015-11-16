require 'dp'
require 'cunn'
require 'nntrain.volume3d'
local create = require 'nntrain.create_dcnn'
local train = require 'nntrain.train'
require 'util.celebrate'


cmd = torch.CmdLine()
cmd:option('--kernelSize', '{13, 13, 13}') 
cmd:option('--nout', '{120, 200, 1}', 'Number of the output feature maps') 
cmd:option('--padding', false)
cmd:option('--datapath', '/home/siqi/hpc-data1/SQ-Workspace/RivuletJournalData/OP/OP1/OP1Feat')
cmd:option('--maxEpoch', 100)
cmd:option('--print_every', 1)
cmd:option('--savemodel', true)
cmd:option('--savemodelprefix', 'op1')
cmd:option('--accUpdate', true, 'accumulate gradients inplace')
cmd:option('--momentum', 1.2, 'momentum')
cmd:option('--learningRate', 0.01, 'learning rate at t=0')
cmd:option('--maxOutNorm', 1, 'max norm each layers output neuron weights')
cmd:option('--silent', false)
cmd:option('--batchsize', 50000)
cmd:option('--nbatch', 17)
cmd:option('--optimization', 'CG')
cmd:option('--maxIter', 10)
cmd:option('--threads', 4)
cmd:option('--visualize', true, 'visualize input data and weights during training')
cmd:option('--weightDecay', 1e-4, 'weight decay')
cmd:option('--learningRateDecay', 1e-3, 'learningRateDecay')
-- cmd:option('--checkgrad', 0, 'Whether check gradients before training')

opt = cmd:parse(arg or {})
opt.kernelSize = table.fromString(opt.kernelSize)
opt.nout = table.fromString(opt.nout)
print(opt)

torch.setnumthreads(opt.threads)
local ds = volume3d(opt)
dcnn, crit = create_dcnn(opt)
train(ds, dcnn, crit, opt)

-- local trainer = nn.StochasticGradient(dcnn, crit)
-- trainer.learningRate = opt.learningRate

-- -- Make dataset
-- dataset={};

-- function dataset:size() return opt.batchsize end -- 100 examples

-- local ds = ds:loadbatch(opt,1)

-- for i=1, dataset:size() do 
--   -- local input = ds.inputs;     -- normally distributed example in 2d
--   -- local output = ds.targets;
--   dataset[i] = {ds.inputs[i], ds.targets[i]}
-- end

-- trainer:train(dataset)

celebrate()
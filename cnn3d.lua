require 'dp'
require 'nntrain.volume3d'
local create = require 'nntrain.create_dcnn'
local train = require 'nntrain.train'
require 'util.celebrate'


cmd = torch.CmdLine()
cmd:option('--kernelSize', '{13, 13, 13}') 
cmd:option('--nout', '{200, 200, 1}', 'Number of the output feature maps') 
cmd:option('--padding', false)
cmd:option('--datapath', '/home/siqi/hpc-data1/SQ-Workspace/RivuletJournalData/OP/OP1/OP1Feat')
cmd:option('--maxEpoch', 10)
cmd:option('--print_every', 1)
cmd:option('--savemodel', true)
cmd:option('--savemodelprefix', 'op1')
cmd:option('--accUpdate', true, 'accumulate gradients inplace')
cmd:option('--momentum', 2, 'momentum')
cmd:option('--learningRate', 0.01, 'learning rate at t=0')
cmd:option('--maxOutNorm', 1, 'max norm each layers output neuron weights')
cmd:option('--silent', false)
cmd:option('--batchsize', 40000)
cmd:option('--nbatch', 10)
cmd:option('--optimization', 'SGD')
cmd:option('--maxIter', 5)
cmd:option('--threads', 4)
cmd:option('--visualize', true, 'visualize input data and weights during training')

opt = cmd:parse(arg or {})
opt.kernelSize = table.fromString(opt.kernelSize)
opt.nout = table.fromString(opt.nout)

torch.setnumthreads(opt.threads)
local ds = volume3d(opt)
dcnn, crit = create_dcnn(opt)
train(ds, dcnn, crit, opt)

celebrate()
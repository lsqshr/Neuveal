-- Assume each subdirectory in the given root direcotory contains all the block batches named as *blocks.mat 
-- and a raw.mat with the raw image voxels for testing

require 'torch'
require 'paths'
require 'dp'
require 'nntrain.volume3d'
require 'nntrain.predictimg'

local create = require 'nntrain.create_dcnn'
local train = require 'nntrain.train'
require 'util.celebrate'

cmd = torch.CmdLine()
cmd:option('--datapath', '/home/siqi/hpc-data1/SQ-Workspace/Neuveal/data/OP')
cmd:option('--kernelSize', '{13, 13, 13}') 
cmd:option('--nout', '{120, 200, 1}', 'Number of the output feature maps') 
cmd:option('--padding', false)
cmd:option('--maxEpoch', 1)
cmd:option('--print_every', 1)
cmd:option('--savemodel', true)
cmd:option('--savemodelprefix', 'op1')
cmd:option('--accUpdate', true, 'accumulate gradients inplace')
cmd:option('--momentum', 1.2, 'momentum')
cmd:option('--learningRate', 0.01, 'learning rate at t=0')
cmd:option('--maxOutNorm', 1, 'max norm each layers output neuron weights')
cmd:option('--silent', false)
cmd:option('--batchsize', 1000)
cmd:option('--optimization', 'CG')
cmd:option('--maxIter', 10)
cmd:option('--threads', 4)
cmd:option('--visualize', true, 'visualize input data and weights during training')
cmd:option('--weightDecay', 1e-4, 'weight decay')
cmd:option('--learningRateDecay', 1e-3, 'learningRateDecay')

opt = cmd:parse(arg or {})
opt.kernelSize = table.fromString(opt.kernelSize)
opt.nout = table.fromString(opt.nout)
print(opt)

torch.setnumthreads(opt.threads)
local dcnn, crit = create_dcnn(opt) -- Create the CNN architecture
local casedirs = paths.dir(opt.datapath) -- list the directories in the given path
local ncase = #casedirs

for i = 1, ncase do 
	-- Sort the cases to perform leave-one-out validation
	local trainblock = {}
	local traingt = {}
	local traincoord = {}

    local trainctr = 0
	for j = 1, ncase do
		-- Make the list of training batch files 
		if i ~= j then
			d = casedirs[j]
			-- print(string.format('Searching %s', paths.concat(opt.datapath, d)))
			local blockfiles = paths.files(paths.concat(opt.datapath, d), 'blocks*')
            for f in blockfiles do
				trainblock[trainctr] = paths.concat(opt.datapath, d, f)
				traingt[trainctr] = paths.concat(opt.datapath, d, 'gt' .. string.sub(f, 7))
				traincoord[trainctr] = paths.concat(opt.datapath, d, 'coord' .. string.sub(f, 7))
			    trainctr = trainctr + 1
			end
		end
	end

	-- Start batch training for this fold
	local ds = volume3d(opt, trainblock, traingt, traincoord)
	model, loss = train(ds, dcnn, crit, opt)
    img = predictimg(paths.concat(opt.datapath, d, 'raw.mat'), model, opt.kernelSize)

	-- Save the models and statistics
    mattorch.save(img, 'data/predict_img_' .. i .. '.mat')
    mattorch.save(model, 'data/predict_model_' .. i .. '.mat')
end
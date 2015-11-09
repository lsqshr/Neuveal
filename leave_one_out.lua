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
cmd:option('--datapath', '/home/siqi/data/OP-sub')
cmd:option('--kernelSize', '{13, 13, 13}') 
cmd:option('--nout', '{30, 200, 1}', 'Number of the output feature maps') 
cmd:option('--padding', false)
cmd:option('--maxEpoch', 15)
cmd:option('--print_every', 1)
cmd:option('--savemodel', true)
cmd:option('--savemodelprefix', 'op-sub')
cmd:option('--accUpdate', true, 'accumulate gradients inplace')
cmd:option('--momentum', 1.2, 'momentum')
cmd:option('--learningRate', 0.01, 'learning rate at t=0')
cmd:option('--maxOutNorm', 1, 'max norm each layers output neuron weights')
cmd:option('--silent', false)
cmd:option('--optimization', 'CG')
cmd:option('--maxIter', 10)
cmd:option('--threads', 8)
cmd:option('--visualize', true, 'visualize input data and weights during training')
cmd:option('--weightDecay', 1e-4, 'weight decay')
cmd:option('--learningRateDecay', 1e-3, 'learningRateDecay')
cmd:option('--singlefold', 1)
cmd:option('--foldidx', 1)
cmd:option('--outdir', 'data/op-sub-30')
cmd:option('--plotfilename', 'plot.html')

opt = cmd:parse(arg or {})
opt.kernelSize = table.fromString(opt.kernelSize)
opt.nout = table.fromString(opt.nout)
-- print(opt)

--Create the outdir if it does not exist
paths.mkdir(opt.outdir)

torch.setnumthreads(opt.threads)
local dcnn, crit = create_dcnn(opt) -- Create the CNN architecture
local casedirs = paths.dir(opt.datapath) -- list the directories in the given path
for i = 1, #casedirs do -- Remove .. and .
    if casedirs[i] == '.' or casedirs[i] == '..' then
    	table.remove(casedirs, i)
    end
end
local ncase = #casedirs

for i = 1, ncase do 

	if (opt.singlefold and opt.foldidx == i)  or opt.singlefold ~= true then

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
				local blockfiles = paths.files(paths.concat(opt.datapath, d, 't7'), 'blocks*')
	            for f in blockfiles do
					startidx, endidx = string.find(f, 'blocks')
					trainblock[trainctr] = paths.concat(opt.datapath, d, 't7', f)
					traingt[trainctr] = paths.concat(opt.datapath, d, 't7', 'gt' .. string.sub(f, endidx + 1))
					traincoord[trainctr] = paths.concat(opt.datapath, d, 't7', 'coord' .. string.sub(f, endidx + 1))
				    trainctr = trainctr + 1
				end
			end
		end

		-- Start batch training for this fold
		local ds = volume3d(opt, trainblock, traingt, traincoord)
		model, loss = train(ds, dcnn, crit, opt)
		fold = {}
		case2work = casedirs[i]
		print('working on ' .. case2work)
	    predimg = predictimg(paths.concat(opt.datapath, case2work, 'raw.mat'), model, opt.kernelSize, 100)
		fold.img = predimg
		fold.model = model

		-- Save the models and statistics
	    torch.save(paths.concat('data', opt.outdir, 'model_fold_' .. i .. '_' .. case2work .. '.t7', fold))
	end
end
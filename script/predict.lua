require 'torch'
require 'dp'
require 'nn'
require 'nntrain.volume3d'
require 'itorch'
require 'mattorch'

cmd = torch.CmdLine()
cmd:option('--modelpath', '/home/siqi/hpc-data1/SQ-Workspace/Neuveal/op1_iter_10.t7')
cmd:option('--datapath', '/home/siqi/hpc-data1/SQ-Workspace/RivuletJournalData/OP/OP1/OP1Feat')
cmd:option('--savepath', '/home/siqi/hpc-data1/SQ-Workspace/RivuletJournalData/OP/OP1')
cmd:option('--batchsize', 50000)
cmd:option('--nbatch', 17)
cmd:option('--imagesize', '{600,600,100}') 

opt = cmd:parse(arg or {})
opt.imagesize = table.fromString(opt.imagesize)

function predict(opt)
	model = torch.load(opt.modelpath)
	ds = volume3d(opt)
	-- pred = torch.DoubleTensor()
	predimg = torch.DoubleTensor(opt.imagesize[1], opt.imagesize[2], opt.imagesize[3])

	for b = 1, opt.nbatch do
		print(string.format("Predicting batch %d/%d\n", b, opt.nbatch))
		local data = ds:loadbatch(opt, b)
		local p = torch.DoubleTensor(data.inputs:size(1))
		for i = 1, data.inputs:size(1) do
			p[i] = model:forward(data.inputs[i])
		end
        -- pred[b] = p

        for cidx=1,data.coord:size(1) do
	        local c = data.coord[cidx]
	        local x = c[{1}]
	        local y = c[{2}]
	        local z = c[{3}]
			print(string.format("Putting x:%d, y:%d, z:%d\n", x, y, z))
	        predimg[{{x}, {y}, {z}}] = p[cidx]
	    end
	end

	return predimg
end


local predimg = predict(opt)

-- maxxy = predimg:max(3)
-- itorch.Image(maxxy)

-- mattorch.save(string.format('%s/pred-scores.mat', opt.savepath), pred)
mattorch.save(string.format('%s/pred-image.mat', opt.savepath), predimg)


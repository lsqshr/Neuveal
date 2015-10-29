require 'torch'
require 'hdf5'
require 'dp'

cmd = torch.CmdLine()
cmd:option('--model', '/home/siqi/hpc-data1/SQ-Workspace/Neuveal/fuck_iter_10.t7')
cmd:option('--data', '/home/siqi/hpc-data1/Data/OP/OP_V3Draw/whole-op-img.h5')
cmd:option('--target', '/home/siqi/hpc-data1/Data/OP/OP_V3Draw/op-train-recon.h5')

opt = cmd:parse(arg or {})

function predict(modelpath, datapath, targetpath)
	local imgf = hdf5.open(datapath)
	imgdata = imgf:read('/op/img'):all():double()
	imgdata = imgdata:permute(4, 3, 2, 1)
	imgdata:resize((#imgdata)[1], 1, (#imgdata)[2], (#imgdata)[3], (#imgdata)[4])

	local model = torch.load(modelpath)
	local recon = model:forward(imgdata[{{5}, {}, {}, {}, {}}])
	-- print(#recon)
	-- print(recon)
	return recon
end

recon = predict(opt.model, opt.data, opt.target)
local targetfile = hdf5.open(opt.target, 'w')
targetfile:write('/op/recon', recon)
targetfile:close()

require 'nntrain.predictimg'
require 'torch'

cmd = torch.CmdLine()
cmd:option('--filepath', 'data/OP/OP6/t7/raw.t7')
cmd:option('--modelpath', 'op1_iter_1.t7')
cmd:option('--kernelSize', '{13, 13, 13}') 
cmd:option('--batchSize', 500)
cmd:option('--outimg', 'predimg.t7')

opt = cmd:parse(arg or {})
opt.kernelSize = table.fromString(opt.kernelSize)
-- opt.nout = table.fromString(opt.nout)

model = torch.load(opt.modelpath)
pimg = predictimg(opt.filepath, model, opt.kernelSize, opt.batchSize)
torch.save(opt.outimg, pimg)
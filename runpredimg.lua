require 'nntrain.predictimg'
require 'torch'

cmd = torch.CmdLine()
cmd:option('--filepath', 'data/OP/OP6/t7/raw.t7')
cmd:option('--modelpath', 'op1_iter_1.t7')
cmd:option('--kernelsize', '{13, 13, 13}') 
cmd:option('--batchsize', 500)
cmd:option('--outimg', 'predimg.t7')
cmd:option('--threads', 4)
cmd:option('--iscuda', 1)

opt = cmd:parse(arg or {})
opt.kernelsize = table.fromString(opt.kernelsize)
-- opt.nout = table.fromString(opt.nout)
-- torch.setnumthreads(opt.threads)
torch.setdefaulttensortype('torch.FloatTensor') 

model = torch.load(opt.modelpath)

pimg = predictimg(opt, model)
torch.save(opt.outimg, pimg)
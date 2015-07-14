require 'dp'
require 'nn'
require 'hdf5'

cmd = torch.CmdLine()
cmd:option('--kernelSize', '{7,7,7}', 'kernel size of each convolution layer. height = width = depth') 
cmd:option('--NOutput', '{3,3, 2000}', 'number of the output feature maps') 
opt = cmd:parse(arg or {})
opt.kernelSize = table.fromString(opt.kernelSize)
opt.NOutput = table.fromString(opt.NOutput)

print('Loading data')
local f = hdf5.open('/home/siqi/hpc-data1/OP_V3Draw/OP_V3Draw-block/whole-op-img.h5')
local imgdata = f:read('/op/img'):all():double()
imgdata = imgdata:permute(4, 3, 2, 1)
imgdata:resize((#imgdata)[1], 1, (#imgdata)[2], (#imgdata)[3], (#imgdata)[4])
local f = hdf5.open('/home/siqi/hpc-data1/OP_V3Draw/OP_V3Draw-block/whole-op-syn.h5')
local syndata = f:read('/op/syn'):all():double()
syndata = syndata:permute(4, 3, 2, 1)

print('Building the first cnn')
print(opt.kernelSize[1], opt.kernelSize[2], opt.kernelSize[3])
print(opt.NOutput[1], opt.NOutput[2])
local cnn1 = nn.Sequential()
-- Convolutional and pooling layers
-- TODO: Dropout
cnn1:add(nn.VolumetricConvolution(1, opt.NOutput[1], opt.kernelSize[1], opt.kernelSize[1], opt.kernelSize[1]))
cnn1:add(nn.VolumetricConvolution(opt.NOutput[1], opt.NOutput[2], opt.kernelSize[2], opt.kernelSize[2], opt.kernelSize[2]))
cnn1:add(nn.VolumetricConvolution(opt.NOutput[2], 1, opt.kernelSize[3], opt.kernelSize[3], opt.kernelSize[3]))
print('get outside')
print((#imgdata)[1], (#imgdata)[2], (#imgdata)[3], (#imgdata)[4])
cnnoutsize1 = cnn1:outside({1, (#imgdata)[3], (#imgdata)[4], (#imgdata)[5]}) 
print('get input size for dense layer')
print(cnnoutsize1[1], cnnoutsize1[2], cnnoutsize1[3], cnnoutsize1[4])
inputSize1 = cnnoutsize1[2] * cnnoutsize1[3] * cnnoutsize1[4]
print('inputSize1: ', inputSize1)
print('collapse output convolution layer')
cnn1:add(nn.Collapse(4))
print('add linear layer')
cnn1:add(nn.Linear(inputSize1, opt.NOutput[3]))
print('add tanh layer')
cnn1:add(nn.Tanh())

print('try forward cnn1')
minibatch = imgdata[{{1,2}, {}, {}, {}, {}}]

print(cnn1:forward(minibatch))

-- print('Building the second cnn')
-- cnn2:add(VolumetricConvolution(1, opt.NOutput[1], opt.kernelSize[1], opt.kernelSize[1]), opt.kernelSize[1])
-- cnn2:add(VolumetricConvolution(opt.NOutput[1], opt.NOutput[2], opt.kernelSize[2], opt.kernelSize[2])
-- cnn2:add(VolumetricConvolution(opt.NOutput[2], 1, opt.kernelSize[3], opt.kernelSize[3]), opt.kernelSize[3])
-- cnnoutsize2 = cnn2:outside{1, imgsz, imgsz, imgsz}
-- inputSize2 = cnnoutsize2[2] * cnnoutsize2[3] * cnnoutsize2[4]
-- cnn2:add(nn.Linear(inputSize2))
-- cnn2:add(nn.Tanh())

-- pr = nn.ParallelTable()
-- pr:add(cnn1)
-- pr:add(cnn2)
-- embed = nn.Sequential()
-- embed:add(pr)
-- embed:add(nn.PairwiseDistance(1))

-- crit = nn.HingeEmbeddingCriterion(1)

-- function gradUpdate(model, x, y, criterion, learningRate)
-- 	local pred = model:forward(x)
-- 	local err = criterion:forward(pred, y)
-- 	local gradCriterion = criterion:backward(pred, y)
-- 	model:zeroGradParameters()
-- 	model:backward(x, gradCriterion)
-- 	model:updateParameters(learningRate)
-- end

-- for i = 1, 1000
-- 	-- TODO: MINI BATCH
-- 	-- for j = 1, NBATCH

-- 	-- end
-- 	gradUpdate(embed, {x, y}, 1, crit, 0.01)
-- 	print(embed:forward(x, y)[1])
-- end

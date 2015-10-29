-- Deprecated
require 'nn'

function create_embed_model(opt, data)
	print('Building the first cnn: cnn1')
	local cnn1 = nn.Sequential()
	-- Convolutional and pooling layers
	print(opt.NOutput, opt.kernelSize)
	print(opt.NOutput[1], opt.kernelSize[1])

	if opt.dropout and (opt.dropoutProb[1] or 0) > 0 then
      -- dropout can be useful for regularization
      cnn1:add(nn.Dropout(opt.dropoutProb[1]))
	end

	cnn1:add(nn.VolumetricConvolution(1, opt.NOutput[1], opt.kernelSize[1], opt.kernelSize[1], opt.kernelSize[1]))

	if opt.dropout and (opt.dropoutProb[2] or 0) > 0 then
      -- dropout can be useful for regularization
      cnn1:add(nn.Dropout(opt.dropoutProb[2]))
	end

	cnn1:add(nn.VolumetricConvolution(opt.NOutput[1], opt.NOutput[2], opt.kernelSize[2], opt.kernelSize[2], opt.kernelSize[2]))

	if opt.dropout and (opt.dropoutProb[3] or 0) > 0 then
      -- dropout can be useful for regularization
      cnn1:add(nn.Dropout(opt.dropoutProb[3]))
	end

	cnn1:add(nn.VolumetricConvolution(opt.NOutput[2], 1, opt.kernelSize[3], opt.kernelSize[3], opt.kernelSize[3]))

	if opt.dropout and (opt.dropoutProb[4] or 0) > 0 then
      -- dropout can be useful for regularization
      cnn1:add(nn.Dropout(opt.dropoutProb[4]))
	end
	
	cnnoutsize1 = cnn1:outside({1, (#data.imgdata)[3], (#data.imgdata)[4], (#data.imgdata)[5]}) 
	inputSize1 = cnnoutsize1[2] * cnnoutsize1[3] * cnnoutsize1[4]
	cnn1:add(nn.Collapse(4))
	cnn1:add(nn.Linear(inputSize1, opt.NOutput[3]))
	cnn1:add(nn.Tanh())

	print('Building the second cnn: cnn2')
	local cnn2 = nn.Sequential()
	-- Convolutional and pooling layers
	if opt.dropout and (opt.dropoutProb[1] or 0) > 0 then
      -- dropout can be useful for regularization
      cnn1:add(nn.Dropout(opt.dropoutProb[1]))
	end

	cnn2:add(nn.VolumetricConvolution(1, opt.NOutput[1], opt.kernelSize[1], opt.kernelSize[1], opt.kernelSize[1]))

	if opt.dropout and (opt.dropoutProb[2] or 0) > 0 then
      -- dropout can be useful for regularization
      cnn1:add(nn.Dropout(opt.dropoutProb[2]))
	end

	cnn2:add(nn.VolumetricConvolution(opt.NOutput[1], opt.NOutput[2], opt.kernelSize[2], opt.kernelSize[2], opt.kernelSize[2]))

	if opt.dropout and (opt.dropoutProb[3] or 0) > 0 then
      -- dropout can be useful for regularization
      cnn1:add(nn.Dropout(opt.dropoutProb[3]))
	end

	cnn2:add(nn.VolumetricConvolution(opt.NOutput[2], 1, opt.kernelSize[3], opt.kernelSize[3], opt.kernelSize[3]))

	if opt.dropout and (opt.dropoutProb[4] or 0) > 0 then
      -- dropout can be useful for regularization
      cnn1:add(nn.Dropout(opt.dropoutProb[4]))
	end
	
	cnnoutsize2 = cnn2:outside({1, (#data.syndata)[3], (#data.syndata)[4], (#data.syndata)[5]}) 
	inputSize2 = cnnoutsize2[2] * cnnoutsize2[3] * cnnoutsize2[4]
	cnn2:add(nn.Collapse(4))
	cnn2:add(nn.Linear(inputSize2, opt.NOutput[3]))
	cnn2:add(nn.Tanh())

	print('Combine two CNNs to parallel Table')
	pr = nn.ParallelTable()
	pr:add(cnn1)
	pr:add(cnn2)
	embed = nn.Sequential()
	embed:add(pr)
	embed:add(nn.PairwiseDistance(1))
	crit = nn.HingeEmbeddingCriterion(1)

	return embed, crit
end

return create_model
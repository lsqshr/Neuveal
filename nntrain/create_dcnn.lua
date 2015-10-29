require 'nn' 

function create_dcnn(opt)
	print('Building the first layer of cnn')

	local dcnn = nn.Sequential()
    dcnn:add(nn.SpatialDropout(0.5))
    dcnn:add(nn.VolumetricConvolution(1, opt.nout[1], 
    	                              opt.kernelSize[1], opt.kernelSize[2],
    	                              opt.kernelSize[3], 1, 1, 1, false))
    -- dcnn:add(nn.SpatialBatchNormalization(1))
    dcnn:add(nn.Sigmoid())
    dcnn:add(nn.Collapse(4))
    dcnn:add(nn.Linear(opt.nout[1], opt.nout[2]))
    dcnn:add(nn.Dropout(0.5))
    -- dcnn:add(nn.BatchNormalization(200))
    dcnn:add(nn.Sigmoid())
    dcnn:add(nn.Linear(200, 1))
    crit = nn.MSECriterion()

    return dcnn, crit
end

return create_dcnn
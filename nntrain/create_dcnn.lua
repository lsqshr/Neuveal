require 'nn' 

function create_dcnn(opt)

	local dcnn = nn.Sequential()
    -- dcnn:add(nn.SpatialDropout(0.5))
    dcnn:add(nn.VolumetricConvolution(1, opt.nout[1], 
    	                              opt.kernelSize[1], opt.kernelSize[2],
    	                              opt.kernelSize[3], 1, 1, 1, false))
    -- dcnn:add(nn.SpatialBatchNormalization(opt.nout[1])) % Seems does not support 5d batch
    dcnn:add(nn.Sigmoid())
    dcnn:add(nn.Collapse(4))
    dcnn:add(nn.Linear(opt.nout[1], opt.nout[2]))
    -- dcnn:add(nn.Normalize(2))
    -- dcnn:add(nn.Dropout(0.5))
    -- dcnn:add(nn.BatchNormalization(opt.nout[2]))
    dcnn:add(nn.Sigmoid())
    dcnn:add(nn.Linear(opt.nout[2], 1))
    crit = nn.MSECriterion()

    return dcnn, crit
end

return create_dcnn
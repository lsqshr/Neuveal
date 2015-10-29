require 'torch'

-- Resample the volumes in format [nvolume, nchannel, x, y, z] using lanczos 
function resample(volumes, outsize, option)

	assert(option.kernelsize % 2 == 1 && option.kernelsize >= 3)

    local volsz = (#volumes)
    local nvol = volsz[1]    
    local targets = torch.Tensor(nvol, 1, volsz[3], volsz[4], volsz[5])

    for i = 1, nvol do

    	-- For each dim

        resample1D(volumes[{i}, {}, {}, {}, {}], target, 1)	




	end

end


local function resample1D(src, dim, outsz, ksize)
	target = torch.Tensor(volsz[3], outsz[2], outsz[3])
	srcsz = (#src)[dim]
	-- Get the other two dims

	if outsize <  -- downsample
    	for p = 1 : outsize
    		-- Find the coresponding position in original image
            local orip = p / outsize * srcsz 

		    -- Make the kernel 
		    shift = (orip - torch.floor(orip)) - 0.5
		    local kx = torch.range( -(ksize - 1)/2, (ksize - 1)/2)
		    kx = kx + shift
		    kw = lanczos(kx)

		    -- Replicate the kernel to 3D and convolve the original image

		    torch.repeatTensor(kw, , 1)

    	end
	else -- Upslcaling
    	for p = 1 : outsize[d]

    	end
    end

    return target
end	

local function lanczos(x)
end
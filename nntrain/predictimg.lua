require 'torch'
require 'dp'
require 'nn'
require 'nntrain.volume3d'
require 'xlua'
-- require 'itorch'
-- require 'mattorch'

function predictimg(filepath, model, kernelsize, batchsize)
    coords = {}
    blocks = torch.zeros(batchsize, 1, kernelsize[1], kernelsize[2], kernelsize[3])

	img = torch.load(filepath)
	-- img = f['img']
	padimg = torch.zeros(img:size(1) + 2 * kernelsize[1],
		                 img:size(2) + 2 * kernelsize[2],
		                 img:size(3) + 2 * kernelsize[3])
	padimg[{{kernelsize[1] + 1, kernelsize[1] + img:size(1)}, 
	       {kernelsize[2] + 1, kernelsize[2] + img:size(2)}, 
	       {kernelsize[3] + 1, kernelsize[3] +  img:size(3)}}] = img;
	predimg = torch.zeros(img:size(1), img:size(2), img:size(3))

	-- pad the image
	kernelradius = {}
    kernelradius[1] = (kernelsize[1] - 1) / 2
    kernelradius[2] = (kernelsize[2] - 1) / 2
    kernelradius[3] = (kernelsize[3] - 1) / 2

    bctr = 0
    pctr = 0
	for x = 1, img:size(1) do
		for y = 1, img:size(2) do
			for z = 1, img:size(3) do
				bctr = bctr + 1
				padx = x + kernelradius[1]
				pady = y + kernelradius[2]
				padz = z + kernelradius[3]

				-- for each location make the prediction
				block = padimg[{{padx - kernelradius[1], padx + kernelradius[1]},
                                {pady - kernelradius[2], pady + kernelradius[2]},
                                {padz - kernelradius[3], padz + kernelradius[3]}}]

			    blocks[{{bctr}, 1, {}, {}, {}}] = block
			    coords[bctr] = {x, y, z}

			    if bctr == batchsize then
			    	print('maxinput: ')
			    	print(maxinput)
				    maxinput = torch.max(blocks)
				    if maxinput > 0 then
				    	-- print('<<<<<<<<<<<<<<<<<<<<<<maxinput: ')
				    	-- print(maxinput)
					    maxmat = torch.Tensor(blocks:size(1), blocks:size(2),
					                          blocks:size(3), blocks:size(4), blocks:size(5)):fill(1/maxinput)
					    blocks = blocks:cmul(maxmat) -- Normalise the inputs
					end

					for i = 1, batchsize do
						predimg[{{coords[i][1]}, {coords[i][2]}, {coords[i][3]}}] = model:forward(blocks[i])
					    xlua.progress(pctr, img:size(1) * img:size(2) * img:size(3))
						pctr = pctr + 1
					end

			    	bctr = 0
				end

			end
		end
	end


	return predimg
end

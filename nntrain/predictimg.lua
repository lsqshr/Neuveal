require 'torch'
require 'dp'
require 'nn'
require 'nntrain.volume3d'
require 'itorch'
require 'mattorch'

function predictimg(filepath, model, kernelsize)
    batchsize = 100
    coords = {}
    blocks = torch.zeros(batchsize, 1, kernelsize[1], kernelsize[2], kernelsize[3])

	f = mattorch.load(filepath)
	img = f['img']
	padimg = torch.zeros(img:size(1) + kernelsize[1] - 1,
		                  img:size(2) + kernelsize[2] - 1,
		                  img:size(3) + kernelsize[3] - 1)
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
				    print(string.format('Predicting %.2f%%', 100 * pctr / (img:size(1) * img:size(2) * img:size(3))))

					pred = model:forward(blocks)

                    for b = 1, bctr do
						predimg[{{coords[b][1]}, {coords[b][2]}, {coords[b][3]}}] = pred[b]
					end

			    	bctr = 0
				end

				pctr = pctr + 1

			end
		end
	end


	return predimg
end

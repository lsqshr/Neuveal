require 'torch'
require 'cutorch'
require 'dp'
require 'nn'
require 'cunn'
require 'nntrain.volume3d'
require 'xlua'
-- require 'itorch'
-- require 'mattorch'

-- function predictimg(opt.filepath, model, opt.batchsize, opt.batchsize, opt.iscuda)
function predictimg(opt, model)
	if opt.iscuda == 1 then
		model = model:cuda()
	end

    coords = {}
    blocks = torch.zeros(opt.batchsize, 1, opt.kernelsize[1], opt.kernelsize[2], opt.kernelsize[3])

	img = torch.load(opt.filepath)
	img = img:float()
	-- img = f['img']
	padimg = torch.zeros(img:size(1) + 2 * opt.kernelsize[1],
		                 img:size(2) + 2 * opt.kernelsize[2],
		                 img:size(3) + 2 * opt.kernelsize[3])
	padimg[{{opt.kernelsize[1] + 1, opt.kernelsize[1] + img:size(1)}, 
	       {opt.kernelsize[2] + 1, opt.kernelsize[2] + img:size(2)}, 
	       {opt.kernelsize[3] + 1, opt.kernelsize[3] +  img:size(3)}}] = img;
	predimg = torch.zeros(img:size(1), img:size(2), img:size(3))

	-- if opt.iscuda then
	--     blocks = blocks:cuda()
	--     predimg = predimg:cuda()
	--     padimg = padimg:cuda()
	-- end

	-- pad the image
	kernelradius = {}
    kernelradius[1] = (opt.kernelsize[1] - 1) / 2
    kernelradius[2] = (opt.kernelsize[2] - 1) / 2
    kernelradius[3] = (opt.kernelsize[3] - 1) / 2

    bctr = 0
    pctr = 0
    nvox = img:size(1) * img:size(2) * img:size(3)
    imgtimer = torch.Timer()
	for x = 1, img:size(1) do
		for y = 1, img:size(2) do
			for z = 1, img:size(3) do
				bctr = bctr + 1
				pctr = pctr + 1
				padx = x + kernelradius[1]
				pady = y + kernelradius[2]
				padz = z + kernelradius[3]

				-- for each location make the prediction
				blocks[{{bctr}, 1, {}, {}, {}}] = padimg[{{padx - kernelradius[1], padx + kernelradius[1]},
                                {pady - kernelradius[2], pady + kernelradius[2]},
                                {padz - kernelradius[3], padz + kernelradius[3]}}]

			    -- blocks[{{bctr}, 1, {}, {}, {}}] = block
			    coords[bctr] = {x, y, z}

			    if bctr == opt.batchsize or nvox == pctr then
				    maxinput = torch.max(blocks)
			    	-- print('maxinput: ')
			    	-- print(maxinput)
				    if maxinput > 0 then
				    	-- print('<<<<<<<<<<<<<<<<<<<<<<maxinput: ')
				    	-- print(maxinput)
					    maxmat = torch.FloatTensor(blocks:size(1), blocks:size(2),
					                          blocks:size(3), blocks:size(4), blocks:size(5)):fill(1/maxinput)
					    -- maxmat = maxmat:cuda()
					    blocks = blocks:cmul(maxmat) -- Normalise the inputs
					end



					batchtimer = torch.Timer()
					if opt.iscuda == 1 then
						print('Start Forwarding -- ')
						result = model:forward(blocks:cuda())
						result = result:float()

						for i = 1, opt.batchsize do
							predimg[{{coords[i][1]}, {coords[i][2]}, {coords[i][3]}}] = result[i]
						end

					else
						for i = 1, opt.batchsize do
							predimg[{{coords[i][1]}, {coords[i][2]}, {coords[i][3]}}] = model:forward(blocks[i])
						    -- xlua.progress(pctr, img:size(1) * img:size(2) * img:size(3))
						end
					end
				    xlua.progress(pctr, nvox)
					print('Time used for 1 block: ' .. 1000 * batchtimer:time().real / opt.batchsize .. ' ms')

			    	bctr = 0
				end

			end
		end
	end

	print(string.format('The prediction of the whole image (%d * %d * %d) took %f s', img:size(1), img:size(2), img:size(3), imgtimer:time().real))

	return predimg
end

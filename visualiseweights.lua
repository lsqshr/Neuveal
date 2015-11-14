require 'torch'
require 'image'
require 'nn'
require 'dp'

cmd = torch.CmdLine()
cmd:option('--modelpath', '/home/siqi/hpc-data1/SQ-Workspace/Neuveal/data/oncompute_120feat_tokyofly/model_fold_1.t7')
opt = cmd:parse(arg or {})

function weightimage(img)

	-- kernelsize = w:size(3)
	-- nkernel = w:size(1)
 --    img = torch.zeros(kernelsize * torch.floor(nkernel / 6), kernelsize * 6)

 --    for i = 1, nkernel do
 --        rowidx = torch.floor((i - 1) / 6)
 --        colidx = i - (rowidx - 1) * 6

 --        img[{{(rowidx - 1) * kernelsize + 1, rowidx * kernelsize}, {(colidx - 1) * kernelsize + 1, colidx * kernelsize}}] = w[i]
 --    end

 --    return img

    -- Adapted from itorch.image
	if torch.isTensor(img) or torch.type(img) == 'table' then
		opts = opts or {padding=2}
		opts.input = img
		local imgDisplay = image.toDisplayTensor(opts)
		if imgDisplay:dim() == 2 then 
			imgDisplay = imgDisplay:view(1, imgDisplay:size(1), imgDisplay:size(2))
			return imgDisplay
		else
			print('Fuck wrong dimension of imgDisplay!')
			return 0 
		end
		-- local tmp = os.tmpname() .. '.png'
		-- image.save(tmp, imgDisplay)
		-- -------------------------------------------------------------
		-- -- load the image back as binary blob
		-- local f = assert(torch.DiskFile(tmp,'r',true)):binary();
		-- f:seekEnd();
		-- local size = f:position()-1
		-- f:seek(1)
		-- local buf = torch.CharStorage(size);
		-- assert(f:readChar(buf) == size, 'wrong number of bytes read')
		-- f:close()
		-- os.execute('rm -f ' .. tmp)
		-- ------------------------------------------------------------
		-- local content = {}
		-- content.source = 'itorch'
		-- content.data = {}
		-- content.data['text/plain'] = 'Console does not support images'
		-- content.data['image/png'] = base64.encode(ffi.string(torch.data(buf), size))
		-- content.metadata = { }
		-- content.metadata['image/png'] = {width = imgDisplay:size(3), height = imgDisplay:size(2)}

		-- local m = util.msg('display_data', itorch._msg)
		-- m.content = content
		-- util.ipyEncodeAndSend(itorch._iopub, m)
		-- else
		-- 	error('unhandled type in itorch.image:' .. torch.type(img))
	end
end

opt = cmd:parse(arg or {})

model = torch.load(opt.modelpath)
weight = model:get(1).weight
weight = torch.max(weight, 3)
weight:resize((#weight)[1], 13, 13)
wimg = weightimage(weight)
image.save(paths.concat(paths.dirname(opt.modelpath), paths.basename(opt.modelpath) .. '.xy.png'), wimg)

weight = model:get(1).weight
weight = torch.max(weight, 4)
weight:resize((#weight)[1], 13, 13)
image.save(paths.concat(paths.dirname(opt.modelpath), paths.basename(opt.modelpath) .. '.yz.png'), wimg)

weight = model:get(1).weight
weight = torch.max(weight, 5)
weight:resize((#weight)[1], 13, 13)
image.save(paths.concat(paths.dirname(opt.modelpath), paths.basename(opt.modelpath) .. '.xz.png'), wimg)

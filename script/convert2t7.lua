-- Convert h5/mat to .t7
require 'torch'
require 'paths'

cmd = torch.CmdLine()
cmd:option('--pathtofile', '')
cmd:option('--pathtodata', '')
opt = cmd:parse(arg or {})

local hdf5 = require 'hdf5'
local imgf = hdf5.open(opt.pathtofile)
imgdata = imgf:read(opt.pathtodata):all():double()
imgdata = imgdata:permute(4, 3, 2, 1)

dir = paths.dirname(opt.pathtofile)
fname = paths.basename(opt.pathtofile)

newpath = dir .. '/' .. fname .. '.t7'
print('saving to ' .. newpath)
torch.save(newpath, imgdata)
-- Convert mat to .t7
-- Assume the matlab file only has one field
require 'torch'
require 'paths'
require 'mattorch'

function tablelength(T)

  local count = 0
  for _ in pairs(T) do count = count + 1 end
  return count
end

cmd = torch.CmdLine()
cmd:option('--path2file', '.')
opt = cmd:parse(arg or {})

dir = paths.dirname(opt.path2file)
fname = paths.basename(opt.path2file, '.mat')

paths.mkdir(paths.concat(dir, 'mat'))
newpath = paths.concat(dir, 'mat', fname .. '.mat')
-- print('saving to ' .. newpath)

tfile = torch.load(opt.path2file)

-- assert(tablelength(mfile) == 1)
l = {img = tfile:double()}

mattorch.save(newpath, l)

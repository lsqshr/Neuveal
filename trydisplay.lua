disp = require 'display' 

require 'image'
require 'socket'


data = {}
for i=1,15 do
    table.insert(data, {i, math.random(), math.random()*2})
end
fuckplot = disp.plot(data, { labels={ 'position', 'a', 'b' }, title='fuck'})

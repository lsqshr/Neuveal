local MseFeedback, parent = torch.class("MseFeedback", "dp.Feedback")

function MseFeedback:__init(config)
   config = config or {}
   assert(torch.type(config) == 'table' and not config[1], 
      "Constructor requires key-value arguments")
   local args, precision, baseline, name = xlua.unpack(
      {config},
      'MseFeedback', 
      'Uses Mean Squared Error to measure the distance between the target and restored image',
      {arg='precision', type='number', req=true,
       help='precision (an integer) of the keypoint coordinates'},
      -- {arg='baseline', type='torch.Tensor', default=false,
      --  help='Constant baseline used for comparison'},
      {arg='name', type='string', default='mse error',
       help='name identifying Feedback in reports'}
   )
   config.name = name
   -- if baseline then
   --    assert(baseline:dim() == 1, "expecting 1D constant-value baseline")
   --    self._baseline = baseline
   --    self._baselineSum = torch.Tensor():zero()
   -- end
   self._precision = precision
   parent.__init(self, config)
   self._output = torch.FloatTensor()
   self._targets = torch.FloatTensor()
   self._sum = torch.Tensor():zero()
   self._count = torch.Tensor():zero()
   self._mse = torch.Tensor()
end


function MseFeedback:setup(config)
   parent.setup(self, config)
   self._mediator:subscribe("doneEpoch", self, "doneEpoch")
end


function MseFeedback:doneEpoch(report)
   if self._n_sample > 0 then
      local msg = self._id:toString().." MSE = "..self:meanSquareError()
      if self._baselineMse then
         msg = msg.." vs "..self._baselineMse
      end
      print(msg)
   end
end


function MseFeedback:_reset()
   self._sum:zero()
   self._count:zero()
end


function MseFeedback:report()
   return { 
      [self:name()] = {
         mse = self._n_sample > 0 and self:meanSquareError() or 0
      },
      n_sample = self._n_sample
   }
end
require 'torch'
require 'math'
require 'nn'
require 'optim'

-- local disp = require 'display'
create = require 'nntrain.create_dcnn'

local function train(datasource, model, criterion, opt)
    ------------------------------------------------------------------------
    -- create model and loss/grad evaluation function
    -- local model, criterion = create(opt)
    if opt.iscuda == 1 then
        model = model:float()
        criterion = criterion:float()
        model = model:cuda()
    end

    local params, grads = model:getParameters()
    collectgarbage()

    nbatch = datasource:get_nbatch()
    print(string.format('nbatch: %d\n', nbatch))

    -- Downsample all the ground truth blocks to the size of the output layer

    -- (re-)initialize weights
    params:uniform(-0.01, 0.01)

    -- ------------------------------------------------------------------------
    -- -- optimization loop
    local losses = {}
    local optim_state = {learningRate = opt.learningRate}

    print('Start Training')
    local epocherrors = {}

    for i = 1, opt.maxEpoch do
        print('epoch: ', i)
        ncase = 0
        time = sys.clock()
        print('<trainer> on training set:')

        local batcherrors={}
        for b = 1, nbatch do
            data = datasource:loadbatch(opt, b)
            local ninput = data.inputs:size(1)
            print("<trainer> online epoch # " .. i .. ' [batchsize = ' .. ninput .. ']')
            local trainerror = 0
            -- Display progress
            xlua.progress(b, nbatch)

            -- return loss, grad
            local feval = function(x)
                if x ~= params then
                    params:copy(x)
                end

                grads:zero()
                local loss = 0

                if opt.iscuda == 1 then
                    data.targets = data.targets:float()
                    data.inputs = data.inputs:float()
                    data.inputs = data.inputs:cuda()
                    local output = model:forward(data.inputs) -- Forward all at once
                    output = output:float()
                    local df = torch.FloatTensor(output:size(1), output:size(2))
                    for t = 1, ninput do
                        local err = criterion:forward(output[t], data.targets[t])
                        loss = loss + err
                        df[t] = criterion:backward(output[t], data.targets[t])
                    end

                    model:backward(data.inputs, df:cuda())
                else
                    for t = 1, ninput do
                        local output = model:forward(data.inputs[t])
                        local err = criterion:forward(output, data.targets[t])
                        loss = loss + err

                        -- backward
                        model:backward(data.inputs[t], criterion:backward(output, data.targets[t]))
                    end
                end

                grads:div(ninput)
                loss = loss/ninput
                trainerror = loss
                -- trainerror = trainerror + loss

                return loss, grads
            end

            -- if opt.checkgrad then
            --     print('Checking Gradients')
            --     optim.checkgrad(feval, params, 1e-9)
            -- end

            -- optimize on current mini-batch
            if opt.optimization == 'CG' then
                config = config or {maxIter = opt.maxIter}
                optim.cg(feval, params, config)
            elseif opt.optimization == 'LBFGS' then
                config = config or {learningRate = opt.learningRate,
                                 maxIter = opt.maxIter,
                                 nCorrection = 10}
                optim.lbfgs(feval, params, config)

            elseif opt.optimization == 'SGD' then
                config = config or {learningRate = opt.learningRate,
                                 weightDecay = opt.weightDecay,
                                 momentum = opt.momentum,
                                 learningRateDecay = opt.learningRateDecay}
                optim.sgd(feval, params, config)

            elseif opt.optimization == 'ASGD' then
                config = config or {eta0 = opt.learningRate,
                                 t0 = nbTrainingPatches * opt.t0}
                _,_,average = optim.asgd(feval, params, config)
            else
                error('unknown optimization method')
            end

            collectgarbage("collect")

            --train error
            -- trainerror = trainerror / ninput 
            if i % opt.print_every == 0 then
                print(string.format("Epoch %4d, Average loss = %.6f", i, trainerror))
            end

            if trainerror == 1/0 or trainerror == 0/0 then
                table.insert(batcherrors, 1)
            else
                table.insert(batcherrors, trainerror)
            end

            -- batchwin = disp.plot(batcherrors, {win=batchwin}) 
            -- plot = itorch.Plot():line(torch.range(1, b), batcherrors, 'blue', 'batch errors'):legend(true):title('Batch Plot'):draw()
            -- plot:save(opt.plotfilename)
            ncase = ncase + ninput

            -- Visualise weights
            -- print(#model:get(2).weight)
            weight = model:get(1).weight
            weight = torch.max(weight, 3)
            weight:resize((#weight)[1], 13, 13)
            -- itorch.image(weight) 

            weight = model:get(1).weight
            weight = torch.max(weight, 4)
            weight:resize((#weight)[1], 13, 13)
            -- itorch.image(weight) 
            
            weight = model:get(1).weight
            weight = torch.max(weight, 5)
            weight:resize((#weight)[1], 13, 13)
            -- itorch.image(weight) 
        end

        if batcherrors:size() > 0 then
            epocherr = torch.mean(torch.Tensor(batcherrors));
        end
        
        if epocherr == 1/0 or epocherr == 0/0 then
            table.insert(epocherrors, 1)
        else
            table.insert(epocherrors, epocherr)
        end

        time = sys.clock() - time
        time = time / ncase
        print("<trainer> time to learn 1 sample = " .. (time*1000) .. 'ms')
        -- epochwin = disp.plot(epocherrors, {win=epochwin}) 
        -- plot = itorch.Plot():line(torch.range(1, i), epocherrors, 'red', 'epoch errors'):legend(true):title('Epoch Plot'):draw()

        if opt.savemodel then
            local cache = {}
            cache.model = model
            cache.opt = opt
            torch.save(paths.concat(opt.outdir, opt.savemodelprefix .. '_iter_' .. tostring(i) .. '.t7'), model) -- Cache the model after every iteration
        end
    end

    return model, losses
end

return train
require 'torch'
require 'math'
require 'nn'
require 'optim'
require 'math'

local disp = require 'display'
create = require 'nntrain.create_dcnn'

local function train(datasource, model, criterion, opt)
    ------------------------------------------------------------------------
    -- create model and loss/grad evaluation function
    -- local model, criterion = create(opt)
    local params, grads = model:getParameters()

    -- Downsample all the ground truth blocks to the size of the output layer

    -- (re-)initialize weights
    params:uniform(-0.01, 0.01)

    -- ------------------------------------------------------------------------
    -- -- optimization loop
    local losses = {}
    local optim_state = {learningRate = opt.learningRate}

    print('Start Training')
    local epocherrors = {}
    local epochwin = disp.plot({}, {labels={'epoch', 'trainerror'}, title='train error'}) 
    local batchwin = disp.plot({}, {labels={'batch', 'trainerror'}, title='train error'}) 

    for i = 1, opt.maxEpoch do
        print('epoch: ', i)
        ncase = 0
        time = sys.clock()
        -- do one epoch
        print('<trainer> on training set:')
        print("<trainer> online epoch # " .. i .. ' [batchsize = ' .. opt.batchsize .. ']')

        local batcherrors={}
        for b = 1, opt.nbatch do
            data = datasource:loadbatch(opt, b)
            local ninput = (#data.inputs)[1]
            local trainerror = 0
            -- Display progress
            xlua.progress(b, opt.nbatch)

            -- return loss, grad
            local feval = function(x)
                if x ~= params then
                    params:copy(x)
                end

                grads:zero()
                local loss = 0

                for t = 1, ninput do
                    local output = model:forward(data.inputs[t])
                    local err = criterion:forward(output, data.targets[t])
                    loss = loss + err

                    -- backward
                    local dloss_doutput = criterion:backward(output, data.targets[t])
                    model:backward(data.inputs[t], dloss_doutput)

                    -- if opt.visualize then
                    --     display(data.inputs[i])
                    -- end
                end

                grads:div(ninput)
                loss = loss/ninput
                trainerror = trainerror + loss

                return loss, grads
            end

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
                                 learningRateDecay = 5e-7}
                optim.sgd(feval, params, config)

            elseif opt.optimization == 'ASGD' then
                config = config or {eta0 = opt.learningRate,
                                 t0 = nbTrainingPatches * opt.t0}
                _,_,average = optim.asgd(feval, params, config)

            else
                error('unknown optimization method')
            end


            --train error
            trainerror = trainerror / ninput 
            if i % opt.print_every == 0 then
                print(string.format("Epoch %4d, Average loss = %.6f", i, trainerror))
            end

            if trainerror == 1/0 or trainerror == 0/0 then
                table.insert(batcherrors, {i, 1})
            else
                table.insert(batcherrors, {i, trainerror})
            end

            dispwin = disp.plot(batcherrors, {win=dispwin}) 
            ncase = ncase + ninput
        end

        epocherr = torch.mean(trainerror);
        if epocherr == 1/0 or epocherr == 0/0 then
            table.insert(epocherrors, {i, 1})
        else
            table.insert(epocherrors, {i, epocherr})
        end

        time = sys.clock() - time
        time = time / ncase
        print("<trainer> time to learn 1 sample = " .. (time*1000) .. 'ms')


        if opt.savemodel then
            local cache = {}
            cache.model = model
            cache.opt = opt
            torch.save(opt.savemodelprefix .. '_iter_' .. tostring(i) .. '.t7', model) -- Cache the model after every iteration
        end
    end

    return model, losses
end

return train
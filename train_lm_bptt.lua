--[[
Copyright (c) 2017-present, Facebook, Inc.
All rights reserved.

This source code is licensed under the CC-NC style license found in the
LICENSE file in the root directory of this source tree.
--]]

--------------------------------------------------------------------------------
-- Language Modeling on PTB-Word, PTB-Char, WikitText103, and wikiText2,
-- using mini-batch or full batch gradient descent or adagrad.
-- This is a standalone script which does not depend on rnnlibv3. Instead it
-- uses recurrent modules from cudnn.
--------------------------------------------------------------------------------
require 'math'
require 'nn'
require 'cutorch'
require 'cudnn'
require 'xlua'

local data  = dofile('data.lua')
local utils = dofile('utils.lua')

local cmd = torch.CmdLine()
-- * Dataset parameters
cmd:option('-dset'       , 'ptbc' , 'name of the dataset')
cmd:option('-dpath'      , ''     , 'path where binary files are stored')
-- * Model parameters.
cmd:option('-model'      , 'LSTM' , 'Type of recurrent net: RNN|LSTM|GRU')
cmd:option('-nhid'       , 1024   , 'Number of hidden units per layer')
cmd:option('-nlayer'     , 1      , 'Number of layers')
-- * Optimization parameters.
cmd:option('-lr'         , 20     , 'Initial learning rate.')
cmd:option('-lrshrink'   , 2      , 'learning rate anneal factor')
cmd:option('-clip'       , 0.5    , 'Gradient clipping')
cmd:option('-maxepoch'   , 100    , 'Upper epoch limit')
cmd:option('-batchsize'  , 256    , 'Batch size per worker')
cmd:option('-update_freq', 1      , 'How often to update parameters.')
cmd:option('-full_batch' , false  , 'Whether to do full batch GD.')
cmd:option('-bptt'       , 75     , 'Sequence length')
cmd:option('-tol'        , 0      , 'number of validation error bumps allowed')
cmd:option('-optim'      , 'sgd'  , 'opt alg: sgd|adagrad|rmsprop|adam|adadelta')
cmd:option('-nmachines'  , 1      , 'number of machines to run batches on')
cmd:option('-machineid'  , 1      , 'unique machine id in the worker pool')
-- * Device parameters.
cmd:option('-devid'      , 1      , 'GPU device id')
cmd:option('-seed'       , 1111   , 'Random seed')
-- * Misc parameters.
cmd:option('-verbose'    , true   , 'be verbose or not')
cmd:option('-reportint'  , 1000   , 'Report interval.')
cmd:option('-datapar'    , false  , 'whether to use data parallel')
local config = cmd:parse(arg)

-- Set the random seed manually for reproducibility.
torch.manualSeed(config.seed)
-- If the GPU is enabled, do some plumbing.
local usegpu = config.devid > 0
if usegpu then
    require 'cutorch'
    require 'cunn'
    cutorch.setDevice (config.devid)
    cutorch.manualSeed(config.seed)
end
if config.cudnn then
    assert(usegpu, "Please specify the device id.")
    require 'cudnn'
end

-- create the directory where check points will be saved
config.savedir = utils.setSaveDir(config)
if not paths.dirp(config.savedir) then
    os.execute('mkdir -p ' .. config.savedir)
end
print(config)

--------------------------------------------------------------------------------
-- LOAD DATA
--------------------------------------------------------------------------------
local batches
if config.dpath == '' then
    config.train, config.valid, config.test = data.setPaths(config.dset)
    local destdir = '/tmp/rnnlm_datasets/torch/' .. config.dset
    if not paths.dirp(destdir) then os.execute('mkdir -p ' .. destdir) end
    batches = data.loadData(config, destdir)
else
    batches = data.loadData(config, config.dpath)
end
local dict  = batches.dict
local train = batches.train
local valid = batches.valid
local test  = batches.test
io.write(string.format('[[ Dictionary size: %10d ]]\n', #dict.idx2word))
io.write(string.format('[[  Train set size: %10d ]]\n', train:size(1)))
io.write(string.format('[[  Valid set size: %10d ]]\n', valid:size(1)))
io.write(string.format('[[   Test set size: %10d ]]\n', test:size(1)))

local bptt      = config.bptt
local bsz       = config.batchsize
local nmachines = config.nmachines
local mid       = config.machineid

-- Ensure that valid and test are divisible by bsz * bptt.
local validbsz, testbsz = 10, 10
local validbptt, testbptt = 1, 1
local valid_nbatch = math.floor(valid:size(1) / (validbptt * validbsz))
valid = valid:narrow(1, 1, valid_nbatch * validbptt * validbsz)
local test_nbatch = math.floor(test:size(1) / (testbptt * testbsz))
test = test:narrow(1, 1, test_nbatch * testbptt * testbsz)

-- Divide up into batches.
valid = valid:view(validbsz, -1):t():contiguous()
test  = test :view(testbsz,  -1):t():contiguous()

collectgarbage()

--------------------------------------------------------------------------------
-- MAKE MODEL
--------------------------------------------------------------------------------
-- override the UpdateParameters method of nn.Module depending on the
-- chosen optimization algorithm
utils.setUpdateParametersMethod(config.optim)
local initrange = 0.1
local nhid      = config.nhid
local nlayer    = config.nlayer
-- Size of the vocabulary.
local ntoken    = #dict.idx2word

-- input lookup table
local lut = nn.LookupTable(ntoken, nhid)
-- recurrent layer
local rec
if config.model == 'RNN' then
    rec = cudnn.RNNTanh(nhid, nhid, nlayer, false, 0, true)
elseif config.model == 'LSTM' then
    rec = cudnn.LSTM(nhid, nhid, nlayer, false, 0, true)
end
-- decoder
local dec = nn.Sequential()
dec:add(nn.View(-1, nhid))
dec:add(nn.Linear(nhid, ntoken))
dec:add(nn.LogSoftMax())
-- full network
local model = nn.Sequential()
model:add(lut)

local model_dp = nn.Sequential()
model_dp:add(rec)
model_dp:add(dec)
if config.datapar == true then
    model_dp = nn.DataParallelTable(1, true, true):add(
        model_dp, torch.range(1, cutorch.getDeviceCount()):totable())
end
model:add(model_dp)

-- initialize parameters
lut.weight:uniform(-initrange, initrange)
dec:get(2).weight:uniform(-initrange, initrange)
dec:get(2).bias:fill(0)

local criterion = nn.CrossEntropyCriterion()

-- If the GPU is enabled, move everything to GPU.
if usegpu then
    model:cuda()
    criterion:cuda()
end

local params, gradParams = model:parameters()
print(model)

--------------------------------------------------------------------------------
-- TRAINING
--------------------------------------------------------------------------------
local timer = torch.Timer()
local dist_timer = torch.Timer()

-- Put the table values into the register file for faster access.
local lr   = config.lr
local clip = config.clip

-- Create buffers for all tensors needed during training on GPU.
local gpubuffer
if usegpu then
    gpubuffer = torch.CudaTensor()
end

-- | Indexes into a dataset and grabs the input and target tokens.
-- Minimizes gpu memory.
local function getExample(data, i, bptt, gpubuffer)
    local newinput, newtarget
    if gpubuffer then
        gpubuffer
            :resize(bptt+1, data:size(2))
            :copy(data:narrow( 1, i, bptt+1 ))
        newinput  = gpubuffer:narrow(1, 1, bptt)
        newtarget = gpubuffer:narrow(1, 2, bptt)
    else
        newinput  = data:narrow( 1 , i    , bptt)
        newtarget = data:narrow( 1 , i + 1, bptt)
    end
    return newinput, newtarget:view(-1)
end


-- | Indexes into a dataset and grabs the input and target tokens
-- corresponding to a full batch for the i-th machine. The function
-- returns a 2D tensor of size (k x bptt) X batchsize, where 'k' is
-- the number of mini-batches assigned to the current worker. In
-- particular ntokens_per_worker = k x bptt x batchsize
local function getDataForWorker(data, macid, bsz, bptt, nmachines)
    assert(macid <= nmachines)
    local newinput, newtarget
    local ntokens = data:nElement()
    local ntokens_per_worker = math.floor(ntokens / nmachines)
    local start = ntokens_per_worker * (macid - 1) + 1
    local size = math.min(ntokens_per_worker, ntokens - start + 1)
    local newdata = data:narrow(1, start, size)
    local nbatches = math.floor(newdata:size(1) / bptt / bsz)
    newdata = newdata:narrow(1, 1, nbatches * bptt * bsz)
    newdata = newdata:view(bsz, -1):t():contiguous()
    return newdata -- (k x bptt) X batchsize
end


-- return the norm of the gradients. used primarily for debugging purpose.
local function gradientNorm(grads)
    local totalnorm = 0
    for mm = 1, #grads do
        local modulenorm = grads[mm]:norm()
        totalnorm = totalnorm + modulenorm * modulenorm
    end
    totalnorm = math.sqrt(totalnorm)
    return totalnorm
end

-- | Gradient clipping to try to prevent the gradient from exploding.
-- Does a global in-place clip.
-- grads : The table of model gradients from :parameters().
-- norm  : The max gradient norm we rescale to.
local function clipGradients(grads, norm)
    local totalnorm = gradientNorm(grads)
    if totalnorm > norm then
        local coeff = norm / math.max(totalnorm, 1e-6)
        for mm = 1, #grads do
            grads[mm]:mul(coeff)
        end
    end
end


-- divide the gradients by n
local function avgGradients(grads, n)
    for mm = 1, #grads do
        grads[mm]:div(n)
    end
end


-- | Perform the forward pass only.
-- model     : The nn.Module.
-- data      : The data tensor (bsz, -1).
-- bptt      : The sequence length (number).
-- bsz       : The size of the batch (number).
-- criterion : The nn.CrossEntropyCriterion module.
local function evaluate(model, data, bptt, bsz, criterion)
    model:evaluate()
    local loss = 0
    local numexamples = 0
    rec:resetStates()
    -- Loop over validation data.
    for i = 1, data:size(1) - bptt, bptt do
        local input, target = getExample(data, i, bptt, gpubuffer)
        loss = loss + criterion:forward(model:forward(input), target)
        numexamples = numexamples + 1
    end
    -- Average out the loss.
    return loss / numexamples
end

-- Loop over epochs.

-- get the input and target
local wdata = getDataForWorker(train, mid, bsz, bptt, nmachines)
local prev_valloss, shrink_ctr
for epoch = 1, config.maxepoch do
    local trainloss = 0
    local nchunks = 0
    timer:reset()
    model:zeroGradParameters()
    -- Loop over the training data.
    model:training()
    ------------------------------------------------------------
    -- Reset the hidden states
    rec:resetStates()
    -- Loop over the data for current worker
    for i = 1, wdata:size(1) - bptt, bptt do
        local tm = torch.Timer()
        local input, target = getExample(wdata, i, bptt, gpubuffer)
        local data_time = tm:time().real * 1000; tm:reset()
        -- forward prop and back prop
        trainloss = trainloss + criterion:forward(model:forward(input), target)
        model:backward(input, criterion:backward(model.output, target))
        nchunks = nchunks + 1
        -- cutorch.synchronize()
        local prop_time = tm:time().real * 1000; tm:reset()

        local comm_time = 0
        if not config.full_batch then
            if nchunks % config.update_freq == 0 then
                comm_time = tm:time().real * 1000; tm:reset()
                avgGradients(gradParams, config.update_freq * nmachines)
                -- clip the gradients
                clipGradients(gradParams, clip)
                -- update the model parameters
                model:updateParameters(lr)
                model:zeroGradParameters()
            end
        end
        -- cutorch.synchronize()
        local optim_time = tm:time().real * 1000; tm:reset()

        if nchunks % config.reportint == 0 then
            local numexamples = nchunks * bptt * bsz
            utils.printf(utils.trainLogStr(
                              epoch, numexamples, lr, timer:time().real, 0,
                              trainloss / nchunks, config.dset))
        end
        -- verbose
        if config.verbose == true then
            utils.printf('[[ %4d ][ %6d / %6d ]] Loss: %3.4f '
                              .. 'Total: %4.0fms Data: %4.0fms '
                              .. 'Prop: %4.0fms Comm: %4.0fms Optim: %4.0fms',
                          mid, i, (wdata:size(1) - bptt), trainloss / nchunks,
                          data_time + prop_time + comm_time + optim_time,
                          data_time, prop_time, comm_time, optim_time)
        end
    end

    ------------------------------------------------------------
    -- exchange the gradients among all nodes
    -- divide the gradients by total number of mini-batches processed
    avgGradients(gradParams, nchunks * nmachines)
    -- clip the gradients
    clipGradients(gradParams, clip)
    -- update the model parameters
    model:updateParameters(lr)

    utils.printf(utils.trainLogStr(
                      epoch, nchunks, lr, timer:time().real,
                      dist_timer:time().real,
                      trainloss / nchunks, config.dset))

    -- evaluate model on validation set
    local valloss = evaluate(model, valid, validbptt, validbsz, criterion)
    utils.printf(utils.validLogStr(epoch, valloss, config.dset))

    ---- learning rate annealing stuff ----
    -- anneal lr only for sgd
    if config.optim == 'sgd' then
        if prev_valloss and prev_valloss <= valloss then
            -- increment the number of times val error increases
            shrink_ctr = shrink_ctr + 1
            -- anneal lr if the number of times val error increases > tolerance
            if shrink_ctr > config.tol then
                shrink_ctr = 0
                lr = lr / config.lrshrink
            end
        else
            shrink_ctr = 0
            prev_valloss = valloss
            -- save the checkpoint
            if mid == 1 and not config.nocheckpoint then
                local timerl = torch.Timer()
                print('saving checkpoint')
                torch.save(paths.concat(config.savedir, 'model.th7'), model)
                print('ended saving checkpoint. Took '
                          .. timerl:time().real .. ' seconds')
            end
        end
        if lr < 1e-5 then break end

    elseif (config.optim == 'adagrad'
                or config.optim == 'rmsprop'
                or config.optim == 'adam'
                or config.optim == 'adadelta') then
        if prev_valloss then
            if prev_valloss[#prev_valloss] > valloss then
                prev_valloss[#prev_valloss + 1] = valloss
                if mid == 1 and not config.nocheckpoint then
                    local timerl = torch.Timer()
                    print('saving checkpoint')
                    torch.save(paths.concat(config.savedir, 'model.th7'), model)
                    print('ended saving checkpoint. Took '
                              .. timerl:time().real .. ' seconds')
                end
            else
                prev_valloss[#prev_valloss + 1] = valloss
                -- check if last tot epochs have not resulted in val_loss decrease
                if #prev_valloss > config.tol then
                    local stop = true
                    for i = 1, config.tol do
                        if (prev_valloss[#prev_valloss - i] >
                            prev_valloss[#prev_valloss - i + 1]) then
                            stop = false
                        end
                    end
                    if stop == true then break end
                end
            end
        else
            prev_valloss = {valloss}
        end
    end
end

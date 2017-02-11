--[[
Copyright (c) 2017-present, Facebook, Inc.
All rights reserved.

This source code is licensed under the CC-NC style license found in the
LICENSE file in the root directory of this source tree.
--]]

require 'math'
require 'nn'
require 'cutorch'
require 'cunn'
require 'cudnn'
require 'nngraph'

local mdls    = dofile('models.lua')
local lutils  = dofile('utils.lua')
local dutils  = dofile('data.lua')

io.stdout:setvbuf('no') -- avoid output buffering

torch.manualSeed(1111)
cutorch.manualSeed(1111)

local tnt    = require 'torchnet'
local json   = require 'cjson'

local cmd = torch.CmdLine()
-- dataset
cmd:option('-dset', 'ptbw', 'name of the dataset to train on')
cmd:option('-dpath', ''   , 'path where binary files are stored')
cmd:option('-train', ''   , 'location of the train file')
cmd:option('-valid', ''   , 'location of the validation file')
cmd:option('-valbsz', 32  , 'minibatch size')
-- model
cmd:option('-model', 'RNN', 'model used: RNN | LSTM | GRU | MRNN')
cmd:option('-nhid', 100, 'number of hidden variables per layer')
cmd:option('-nlayer', 1, 'number of layers in recurrent net')
cmd:option('-block_size', 10, 'numbers of steps to put together')
cmd:option('-init_range', 0.1, 'param (uniform) init range')
cmd:option('-alm', false, 'use ALM rather than ADMM')
-- training
cmd:option('-maxepoch', 10, 'upper epoch limit')
cmd:option('-tol', 5, 'tolerance for early stopping')

cmd:option('-batchsize', 32, 'size of the mini-batch')
cmd:option('-max_ppl', 50000, 'break if we exceed this ppl in training')
cmd:option('-optim', 'adagrad', 'optimization algorithm: gd | adagrad')
-- only if alm=true
cmd:option('-alm_maxsteps', 5, 'number of steps to take on params and hs')
cmd:option('-alm_grad_tol', 1e-5, 'tol for norm of gradient wrt primals')
---- hs
cmd:option('-h_lr', 1, 'eta for minimizing wrt hs')
cmd:option('-h_tol', 1e-5, 'tol for norm of gradient wrt hs')
cmd:option('-h_maxsteps', 10, 'number of descent steps on hs')
cmd:option('-h_clip', 0, 'clip threshold of gradients w.r.t. hs')
---- params
cmd:option('-param_lr', 0.1, 'eta for minimizing wrt recurrent params')
cmd:option('-param_clip', 0.5, 'clip threshold of gradients w.r.t. params')
cmd:option('-param_maxsteps', 5, 'number of steps to take on params')
---- us
cmd:option('-u_lr', 0.1, 'eta for maximizing wrt us')
cmd:option('-u_clip', 0, 'clip threshold of gradients w.r.t. params')
cmd:option('-u_startupdate', 3, 'start updating us on this epoch')
cmd:option('-lambda', 1, 'penalize each l2 sq dist term by lambda/2')
cmd:option('-lambda_mult', 1, 'multiply lambda by this each epoch')
-- others
cmd:option('-eval_on_train', true, 'do train too')
cmd:option('-save_params', false, 'save best params')

cmd:option('-devid', 1, 'master device id +1')
cmd:option('-mb_init', false, 'init each mini-batch w/ fwd preds')
cmd:option('-zero_init', false, 'init each mini-batch w/ 0s')
cmd:option('-opt_arctanh', false, 'optimize z_t s.t. h_t = tanh(z_t)')
cmd:option('-get_positional_ppl', false,
  'get average positional ppl during training')

local config = cmd:parse(arg)

cutorch.setDevice(config.devid)

--------------------------------------------------------------------------------
---- LOAD DATA
--------------------------------------------------------------------------------
local batches
if config.dpath == '' then
    config.train, config.valid, config.test = dutils.setPaths(config.dset)
    local destdir = '/tmp/rnnlm_datasets/torch/' .. config.dset
    if not paths.dirp(destdir) then os.execute('mkdir -p ' .. destdir) end
    batches = dutils.loadData(config, destdir)
else
    batches = dutils.loadData(config, config.dpath)
end
local dict      = batches.dict
local traindata = batches.train
local validdata = batches.valid
local testdata  = batches.test
local ntypes    = #dict.idx2word
io.write(string.format('[[ Dictionary size: %10d ]]\n', #dict.idx2word))
io.write(string.format('[[  Train set size: %10d ]]\n', traindata:size(1)))
io.write(string.format('[[  Valid set size: %10d ]]\n', validdata:size(1)))
io.write(string.format('[[   Test set size: %10d ]]\n', testdata:size(1)))

-- Truncate train and reshape.
local blocksize  = config.block_size
local nblocks  = math.floor((traindata:size(1) - 1) / blocksize)
local train = {}
train.inputs = traindata:narrow(1, 1, nblocks * blocksize)
train.inputs = train.inputs:view(nblocks, -1):t():contiguous()
train.targets = traindata:narrow(1, 2, nblocks * blocksize)
train.targets = train.targets:view(nblocks, -1):t():contiguous()

-- Truncate and reshapre the validation data
local validbsz = config.valbsz
local valid_nbatch = math.floor((validdata:size(1) - 1)/validbsz)
local valid = {}
valid.inputs = validdata:narrow(1, 1, valid_nbatch * validbsz)
valid.inputs = valid.inputs:view(validbsz, -1):t():contiguous()
valid.targets = validdata:narrow(1, 2, valid_nbatch * validbsz)
valid.targets = valid.targets:view(validbsz, -1):t():contiguous()

-- for eval'ing on train
local trainbsz = config.valbsz
local train_nbatch = math.floor((traindata:size(1) - 1)/trainbsz)
local smalltrain = {} -- mini-batched train for evaluating train ppl
smalltrain.inputs = traindata:narrow(1, 1, train_nbatch * trainbsz)
smalltrain.inputs = smalltrain.inputs:view(trainbsz, -1):t():contiguous()
smalltrain.targets = traindata:narrow(1, 2, train_nbatch * trainbsz)
smalltrain.targets = smalltrain.targets:view(trainbsz, -1):t():contiguous()

collectgarbage()
collectgarbage()

--------------------------------------------------------------------------------
-- MAKE MODEL
--------------------------------------------------------------------------------
local nhid   = config.nhid -- no. of hidden units
local nlayer = config.nlayer
-- no. of spring dimension (2*nhid for lstm)
local ndim = config.model == 'LSTM' and 2*nhid or nhid
local block_size = config.block_size
local num_blocks = train.inputs:size(2)
assert(math.ceil(num_blocks) == num_blocks)
local batchsize = config.batchsize

-- model layers
torch.manualSeed(1111)
cutorch.manualSeed(1111)
local lut, recurrentLayer, decoder, gNetwork, trainGraph
if config.model == 'RNN' then
    lut, recurrentLayer, decoder, gNetwork, trainGraph =
        mdls.make_cudnn_rnntanh(nhid, ntypes)
elseif config.model == 'GRU' then
    lut, recurrentLayer, decoder, gNetwork, trainGraph =
        mdls.make_cudnn_gru(nhid, ntypes)
elseif config.model == 'LSTM' then
    lut, recurrentLayer, decoder, gNetwork, trainGraph =
        mdls.make_cudnn_lstm(nhid, ntypes)
end

lut.weight:uniform(-config.init_range, config.init_range)
decoder:get(2).weight:uniform(-config.init_range, config.init_range)
decoder:get(2).bias:fill(0)

-- critierions
local word_weights = torch.ones(ntypes)
word_weights[1] = 0  -- PAD idx
local xentCrit = nn.ClassNLLCriterion(word_weights):cuda()
xentCrit.sizeAverage = false -- will do manually
-- assume same lambda for everybody, which we should maybe change
local mseCrit = nn.MSECriterion():cuda()
mseCrit.sizeAverage = false

local tanhlayer1, tanhlayer2
if config.opt_arctanh then
    tanhlayer1 = nn.Tanh():cuda()
    tanhlayer2 = nn.Tanh():cuda()
end

local params, grad_params = trainGraph:parameters()

local meters = {}
if config.alm then
    meters['xent'] = tnt.AverageValueMeter()
    meters['l2'] = tnt.AverageValueMeter()
    meters['step'] = tnt.AverageValueMeter()
else
    meters['pxent'] = tnt.AverageValueMeter()
    meters['pl2'] = tnt.AverageValueMeter()
    meters['hxent'] = tnt.AverageValueMeter()
    meters['hl2'] = tnt.AverageValueMeter()
    meters['hstep'] = tnt.AverageValueMeter()
end
meters['lag'] = tnt.AverageValueMeter()
-- meters['res'] = tnt.AverageValueMeter()
meters['ul2'] = tnt.AverageValueMeter()

local pos_ppl_meters
if config.get_positional_ppl then
    pos_ppl_meters = {}
    for j = 1, config.block_size do
        pos_ppl_meters[j] = tnt.AverageValueMeter()
    end
end

local function setInitialHiddenState(rnn, states)
    if config.model ~= 'LSTM' then
        if rnn.hiddenInput == nil then
            rnn.hiddenInput = states:cuda()
        else
            rnn.hiddenInput:resize(states:size(1), states:size(2),
                states:size(3)):copy(states)
        end
        rnn.cellInput = nil
    else
        local hid = states:narrow(3, 1, nhid)
        local cel = states:narrow(3, nhid, nhid)
        if rnn.hiddenInput == nil then
            rnn.hiddenInput = hid:cuda()
            rnn.cellInput   = cel:cuda()
        else
            rnn.hiddenInput:resize(hid:size(1), hid:size(2), hid:size(3))
                :copy(hid)
            rnn.cellInput:resize(cel:size(1), cel:size(2), cel:size(3))
                :copy(cel)
        end
    end
end

function minibatch_init_hs(hs, inputs, rnn, lut, last_output)
    assert(config.model ~= 'LSTM')
    -- use last output from previous mini-batch to initialize first block
    hs[1][1]:copy(last_output)
    -- now go forward to get initialization for remaining blocks
    rnn:resetStates()
    rnn:forward(lut:forward(inputs))
    hs[1]:sub(2, hs:size(2))
        :copy(rnn.output[-1]:sub(1, hs:size(2)-1))
    -- the recurrentLayer's hidden states should now be reset to the hs
end

local function scaleGradients(grads, scale)
    for i, v in pairs(grads) do
        v:mul(scale)
    end
end

local function gradsNorm(grads)
    local norm_sum = 0
    for i, v in pairs(grads) do
        norm_sum = norm_sum + v:norm(2)^2
    end
    return torch.sqrt(norm_sum)
end

-- minimizes the the following loss (summed over all blocks), wrt hs and params:
-- \sum_{t=2}^{block_size} KL(decoder(f(x_t, h_{t-1})), y_t)
--     + KL(decoder(h_1), y_1) + lambda/2 ||f(x_1, h_0) - h_1 + u_1||^2,
-- where h_1 represents the first time-step in a block, h0 represents the last
-- timestep in the previous block, and h_t for t > 1 is given by the
-- usual rnn recurrence.
-- parameters of f and decoder are held fixed.
-- h_0 is not optimized.
-- expected tensor sizes:
--  hs and h_grads: 1 x num_blocks x nhid
--  us: (num_blocks-1) x nhid
--  Xs: block_size x num_blocks x nhid
--  targs: block_size x num_blocks
-- other args:
--  opt_state: a (possibly empty) table for the optimizer to store state info
--  meter: a tnt.AverageValueMeter
function optPrimals(ohs, ohst, h_grads, us, inputs, targets, last_output)
    local hs = config.opt_arctanh and tanhlayer1:forward(ohs) or ohs
    local hst = config.opt_arctanh and tanhlayer2:forward(ohst) or ohst
    local nblocks = inputs:size(2)
    -- targets for the two losses.
    local xent_targets = targets:contiguous():view(-1)
    local mse_targets  = hst
    -- set the initial hidden states to the current hs
    setInitialHiddenState(recurrentLayer, hs)
    -- fprop
    local mout = trainGraph:forward({inputs, us})
    -- mout[1]: input to logsoftmax (ntokens x ntypes)
    -- mout[2]: preds + duals (num_blocks x nhid)
    local pos_ppl
    if config.get_positional_ppl then -- inefficient, but just for sanity
        pos_ppl = {}
        local batch_size = targets:size(2)
        for j = 1, targets:size(1) do
            local loss_j = xentCrit:forward(mout[1]:narrow(1,
              (j-1)*batch_size+1, batch_size), targets[j])/batch_size
            pos_ppl[j] = math.pow(2, loss_j/math.log(2))
        end
    end
    local xent_loss = xentCrit:forward(mout[1], xent_targets)
    local l2_loss   = mseCrit:forward(mout[2], mse_targets)
    xent_loss = xent_loss / config.block_size
    l2_loss = l2_loss * config.lambda / 2

    -- backprop through l2 loss
    local dl2df  = mseCrit:backward(mout[2], mse_targets)
    dl2df:mul(config.lambda / 2) -- l2 loss scaled by lambda/2

    -- accumulate the gradients of l2 loss wrt hst (negate b/c second arg)
    if config.opt_arctanh then
        tanhlayer2:backward(ohst, dl2df)
        h_grads[1]:narrow(1, 2, nblocks):add(-1, tanhlayer2.gradInput)
    else
        h_grads[1]:narrow(1, 2, nblocks):add(-1, dl2df)
    end

    -- backprop through the xent loss
    local dxent = xentCrit:backward(mout[1], xent_targets)
    dxent:mul(1 / config.block_size)

    -- now backprop through the network
    trainGraph:backward({inputs, us}, {dxent, dl2df})

    -- gradHiddenInput contains gradients wrt the initial hidden state
    if config.model ~= 'LSTM' then
        local hs_gradin = config.opt_arctanh
          and tanhlayer1:backward(ohs, recurrentLayer.gradHiddenInput)
            or recurrentLayer.gradHiddenInput
        h_grads:narrow(2, 1, nblocks):add(hs_gradin)
    else
        local gradHids = recurrentLayer.gradHiddenInput
        local gradCels = recurrentLayer.gradCellInput
        h_grads:narrow(2, 1, nblocks)
            :add(torch.cat(gradHids, gradCels, 3))
    end

    return {xent = xent_loss, l2 = l2_loss, pos_ppl = pos_ppl}
end

-- minimizes the the following loss (summed over all blocks), wrt all h's:
-- \sum_{t=2}^{block_size} KL(decoder(f(x_t, h_{t-1})), y_t)
--     + KL(decoder(h_1), y_1) + lambda/2 ||f(x_1, h_0) - h_1 + u_1||^2,
-- where h_1 represents the first time-step in a block, h0 represents the last
-- timestep in the previous block, and h_t for t > 1 is given by the
-- usual rnn recurrence.
-- parameters of f and decoder are held fixed.
-- h_0 is not optimized.
-- expected tensor sizes:
--  hs and h_grads: 1 x num_blocks x nhid
--  us: (num_blocks-1) x nhid
--  Xs: block_size x num_blocks x nhid
--  targs: block_size x num_blocks
-- other args:
--  opt_state: a (possibly empty) table for the optimizer to store state info
--  meter: a tnt.AverageValueMeter
function optHiddens(ohs, ohst, h_grads, us, inputs, targets, last_output)
    local hs = config.opt_arctanh and tanhlayer1:forward(ohs) or ohs
    local hst = config.opt_arctanh and tanhlayer2:forward(ohst) or ohst
    local nblocks = inputs:size(2)

    -- targets for the two losses.
    local xent_targets = targets:contiguous():view(-1)
    local mse_targets  = hst

    -- set the initial hidden states to the current hs
    setInitialHiddenState(recurrentLayer, hs)
    -- fprop
    local mout = trainGraph:forward({inputs, us})
    -- mout[1]: input to logsoftmax (ntokens x ntypes)
    -- mout[2]: preds + duals (num_blocks x nhid)

    local xent_loss = xentCrit:forward(mout[1], xent_targets)
    local l2_loss   = mseCrit:forward(mout[2], mse_targets)
    xent_loss = xent_loss / config.block_size
    l2_loss = l2_loss * config.lambda / 2

    -- backprop through l2 loss
    local dl2df  = mseCrit:backward(mout[2], mse_targets)
    dl2df:mul(config.lambda / 2) -- l2 loss scaled by lambda/2

    -- accumulate the gradients of l2 loss wrt hst (negate b/c second arg)
    if config.opt_arctanh then
        tanhlayer2:backward(ohst, dl2df)
        h_grads[1]:narrow(1, 2, nblocks):add(-1, tanhlayer2.gradInput)
    else
        h_grads[1]:narrow(1, 2, nblocks):add(-1, dl2df)
    end

    -- backprop through the xent loss
    local dxent = xentCrit:backward(mout[1], xent_targets)
    dxent:mul(1 / config.block_size)

    -- now backprop through the network
    trainGraph:updateGradInput({inputs, us}, {dxent, dl2df})

    -- gradHiddenInput contains gradients wrt the initial hidden state
    if config.model ~= 'LSTM' then
        local hs_gradin = config.opt_arctanh
          and tanhlayer1:backward(ohs, recurrentLayer.gradHiddenInput)
            or recurrentLayer.gradHiddenInput
        h_grads:narrow(2, 1, nblocks):add(hs_gradin)
    else
        local gradHids = recurrentLayer.gradHiddenInput
        local gradCels = recurrentLayer.gradCellInput
        h_grads:narrow(2, 1, nblocks)
            :add(torch.cat(gradHids, gradCels, 3))
    end

    return {hxent = xent_loss, hl2 = l2_loss}
end

function updateHiddens(hs, h_grads, opt_state)
    if config.h_clip > 0 then
        --print('WARNING: clipping the gradients for hidden states!')
        lutils.clipGradients({h_grads}, config.h_clip)
    end
    if config.optim == 'nag' then
        lutils.nag_step(hs, h_grads, opt_state)
    elseif config.optim == 'adagrad' then
        opt_state = lutils.adagrad_step({hs}, {h_grads},
                                        config.h_lr, opt_state)
    end
end

-- minimizes (for one step) the following loss (summed over all blocks), wrt
-- recurrent parameters, lookup parameters, and decoder parameters.
-- \sum_{t=2}^{block_size} KL(decoder(f(x_t, h_{t-1})), y_t)
--     + KL(decoder(h_1), y_1) + lambda/2 ||f(x_1, h_0) - h_1 + u_1||^2,
-- where h_1 represents the first time-step in a block, h0 represents the last
-- timestep in the previous block, and h_t for t > 1 is given by the
-- usual rnn recurrence.
-- the h's are held fixed.
-- expected tensor sizes:
--  hs: num_blocks x 1 x nhid
--  h0: 1 x nhid
--  us: (num_blocks-1) x 1 x nhid
--  Xs and lut_grad_out: block_size x num_blocks x nhid
--  train and targs: block_size x num_blocks
-- other args:
--  params, grad_params: flattened params and grads associated with
--    lut, clones, and decoder
--  opt_state: a (possibly empty) table for the optimizer to store state info
function optParams(ohs, ohst, us, inputs, targets, params, grad_params)
    local hs = config.opt_arctanh and tanhlayer1:forward(ohs) or ohs
    local hst = config.opt_arctanh and tanhlayer2:forward(ohst) or ohst
    -- targets for the two losses
    local xent_targets = targets:contiguous():view(-1)
    local mse_targets  = hst

    -- set the initial hidden states to the current hs
    setInitialHiddenState(recurrentLayer, hs)
    -- fprop
    local mout = trainGraph:forward({inputs, us})
    -- mout[1]: input to logsoftmax (ntokens x ntypes)
    -- mout[2]: preds + duals (num_blocks x nhid)
    local pos_ppl
    if config.get_positional_ppl then -- inefficient, but just for sanity
        pos_ppl = {}
        local batch_size = targets:size(2)
        for j = 1, targets:size(1) do
            local loss_j = xentCrit:forward(mout[1]:narrow(1,
              (j-1)*batch_size+1, batch_size), targets[j])/batch_size
            pos_ppl[j] = math.pow(2, loss_j/math.log(2))
        end
    end

    local xent_loss = xentCrit:forward(mout[1], xent_targets)
    local l2_loss   = mseCrit:forward(mout[2],  mse_targets)
    xent_loss = xent_loss / config.block_size
    l2_loss   = l2_loss * config.lambda / 2
    -- backprop through l2 loss
    local dl2df  = mseCrit:backward(mout[2], mse_targets)
    dl2df:mul(config.lambda / 2) -- l2 loss scaled by lambda/2
    -- backprop through the xent loss
    local dxent = xentCrit:backward(mout[1], xent_targets)
    dxent:mul(1 / config.block_size)
    -- now backprop through the network
    trainGraph:backward({inputs, us}, {dxent, dl2df})

    return {pxent = xent_loss, pl2 = l2_loss, ppos_ppl = pos_ppl}
end

function updateParams(params, grad_params, opt_state, nblocks)
    -- clip the gradients if specified
    if config.param_clip > 0 then
        lutils.clipGradients(grad_params, config.param_clip)
    end
    -- update the parameters
    if config.optim == 'nag' then
        lutils.nag_step(params, grad_params, opt_state)
    elseif config.optim == 'gd' then
        for i, v in pairs(params) do
            v:add(-config.param_lr, grad_params[i])
        end
    elseif config.optim == 'adagrad' then
        opt_state = lutils.adagrad_step(params, grad_params,
                                        config.param_lr, opt_state)
    end
end

-- MAXIMIZES (for one step) the following loss wrt u's:
-- \sum_{t=2}^{block_size} KL(decoder(f(x_t, h_{t-1})), y_t)
--     + KL(decoder(h_1), y_1) + lambda/2 ||f(x_1, h_0) - h_1 + u_1||^2,
-- where h_1 represents the first time-step in a block, h0 represents the last
-- timestep in the previous block, and h_t for t > 1 is given by the
-- usual rnn recurrence.
-- f's parameters and h's are held fixed.
-- expected tensor sizes:
--       hs: nblocks x nhid
--      hst: nblocks x nhid (targets for the springs)
--       us: (num_blocks-1) x nhid ??
--  u_grads: (num_blocks-1) x nhid ??
--   inputs: block_size x nblocks
--  targets: block_size x nblocks
-- other args:
function optDuals(ohs, ohst, us, u_grads, inputs)
    local hs = config.opt_arctanh and tanhlayer1:forward(ohs) or ohs
    local hst = config.opt_arctanh and tanhlayer2:forward(ohst) or ohst
    local mse_targets = hst[1]
    -- set the initial hidden state to current springs
    setInitialHiddenState(recurrentLayer, hs)
    -- fprop
    local mout = trainGraph:forward({inputs, us})
    local l2_loss = mseCrit:forward(mout[2], mse_targets)
    l2_loss = l2_loss * config.lambda / 2
    -- backprop through l2 loss
    local dl2du = mseCrit:backward(mout[2], mse_targets)
    dl2du:mul(config.lambda / 2)
    -- add (incoming) grads wrt us
    u_grads:copy(dl2du)

    return {ul2 = l2_loss}
end

--        us: nblocks x nhid
--    ugrads: nblocks x nhid
-- opt_state: a (possibly empty) table for the optimizer to store state info
-- the function assumes that hidden states hs have been previously set
function updateDuals(us, ugrads, opt_state)
    opt_state.lr = -config.u_lr --  N.B. MAXIMIZING!!!!
    if config.u_clip > 0 then
        lutils.clipGradients(ugrads, config.u_clip)
    end
    if config.fancy_dual_update then
        if config.optim == 'nag' then
            lutils.nag_step(us, ugrads, opt_state)
        elseif config.optim == 'adagrad' then
            lutils.adagrad_step({us}, {ugrads}, opt_state)
        end
    else
        us:add(config.u_lr, ugrads)
    end
end

-- overwrites h0
function evaluate(data)
    -- set the initial hidden state to current springs
    recurrentLayer:evaluate()
    recurrentLayer:resetStates()
    recurrentLayer.rememberStates = true
    local evalCrit = nn.ClassNLLCriterion(word_weights):cuda()
    evalCrit.sizeAverage = false
    local loss = 0
    local total_steps = 0
    local batch_size  = data.inputs:size(2)
    local sample = {}
    for i = 1, data.inputs:size(1) do
        sample.input = data.inputs:narrow(1, i, 1):cuda()
        sample.target = data.targets:narrow(1, i, 1):view(-1):cuda()
        local hids = gNetwork:forward(sample.input)
        local preds
        if config.model ~= 'LSTM' then
            preds = decoder:forward(hids)
        else
            preds = decoder:forward(hids[1])
        end
        loss = loss + evalCrit:forward(preds, sample.target)
        total_steps = total_steps + 1
    end
    -- reset the states again
    recurrentLayer:resetStates()
    -- -- get avg loss
    loss = loss / (total_steps * batch_size)
    local ppl = math.pow(2, loss/math.log(2))
    recurrentLayer.rememberStates = false
    return ppl
end


function log_stats(d)
   print("json_stats: " .. json.encode(d))
end

local function atanh(x)
    if x >= 1 then
        return 18
    elseif x <= -1 then
        return -18
    else
        return 0.5*torch.log((1+x)/(1-x))
    end
end

function getDataSlice(inputs, targets, hids, ghids, duals, gduals, start, size)
    local rhids = hids:resize(1, size+1, ndim):zero()
    local rghids = ghids:resize(1, size+1, ndim)
    local rduals = duals:resize(1, size, ndim):zero()
    local rgduals = gduals:resize(1, size, ndim)
    return {inputs:narrow(2, start, size):cuda(),
            targets:narrow(2, start, size):cuda(),
            rhids:narrow(2, 1, size),
            rhids:narrow(2, 2, size),
            rghids,
            -- one extra for upstream gradients for last step
            rduals,
            rgduals
    }
end


---- allocate memory for the springs
-- first dimension of hids refer to the number of layers
local hids        = torch.CudaTensor()
local duals       = torch.CudaTensor()
local grad_hids   = hids:clone()
local grad_duals  = duals:clone()
local last_output
if config.mb_init then
    -- stores final hidden state prediction
    last_output = torch.CudaTensor(ndim):zero()
end
-- ignore the last block which is a special case since it does not
-- have any spring at the end
local inputs  = train.inputs:narrow(2, 1, train.inputs:size(2) - 1)
local targets = train.targets:narrow(2, 1, train.targets:size(2) - 1)

local last_good_epoch = 0
-- assume 1 layer for now
function do_training(train, valid, smalltrain)
    local nblocks = inputs:size(2)
    local nbatches = math.ceil(nblocks / batchsize)
    local updates_per_epoch = nbatches
    local p_opt_state, u_opt_state, h_opt_state = {}, {}, {}
    if config.optim == 'nag' then -- put lr in state so we can decay (locally)
        p_opt_state.lr = config.param_lr
        h_opt_state.lr = config.h_lr
        if config.fancy_dual_update then
            u_opt_state.lr = config.u_lr
        end
    end
    local best_trppl, best_vlppl = math.huge, math.huge

    local curr_inputs, curr_targets
    local curr_hid_inputs, curr_hid_targets, curr_grad_hids
    local curr_duals, curr_grad_duals

    for epoch = 1, config.maxepoch do
        print("epoch", epoch)
        lutils.resetMeters(meters)
        if config.get_positional_ppl then
            lutils.resetMeters(pos_ppl_meters)
        end
        trainGraph:training()

        local start_batch, stop_batch = 1, 1
        local u_l2 = 0
        local h_xent, h_l2, p_xent, p_l2 = 0, 0, 0, 0
        local opt_steps = 0
        local blocks_in_update = 0

        for uu = 1, updates_per_epoch do
            blocks_in_update = 0
            h_opt_state = {} -- always restarting h optimization
            -- just get this batch
            local i = start_batch
            if i % 1000 == 0 then
                print("u-batch", i)
            end
            local start = (i - 1) * batchsize + 1
            local size  = math.min(batchsize, nblocks - start + 1)
            blocks_in_update = blocks_in_update + size
            curr_inputs, curr_targets,
            curr_hid_inputs, curr_hid_targets, curr_grad_hids,
            curr_duals, curr_grad_duals =
                unpack(getDataSlice(inputs, targets,
                                    hids, grad_hids,
                                    duals, grad_duals,
                                    start, size))

            if config.zero_init then
                hids:zero()
            elseif config.mb_init then
                minibatch_init_hs(curr_hid_inputs, curr_inputs, recurrentLayer,
                lut, last_output)
                -- hacky, but simpler than alternatives
                if config.opt_arctanh then
                    hids:apply(atanh)
                end
            end

            -- do u's before hs if not initing w/ fwd preds
            if epoch >= config.u_startupdate and not config.mb_init then
                grad_duals:zero()
                local res = optDuals(curr_hid_inputs, curr_hid_targets,
                                     curr_duals, curr_grad_duals,
                                     curr_inputs)
                u_l2 = u_l2 + res['ul2']
                -- scale the gradients and update
                grad_duals:mul(1/blocks_in_update)
                updateDuals(duals, grad_duals, u_opt_state)
            end

            -- optimize wrt primals either alternatingly (admm) or jointly (alm)
            local prevloss, currloss = math.huge, math.huge
            if config.optim == 'gd' then p_opt_state.lr = config.param_lr end
            local nstep = config.alm and config.alm_maxsteps
                or config.h_maxsteps
            opt_steps = 0
            for step = 1, nstep do
                grad_hids:zero()
                trainGraph:zeroGradParameters()
                if config.alm then
                    local res = optPrimals(curr_hid_inputs,
                                      curr_hid_targets, curr_grad_hids,
                                      curr_duals, curr_inputs, curr_targets)
                    h_xent = h_xent + res['xent']
                    h_l2   = h_l2 + res['l2']
                    if config.get_positional_ppl then
                        lutils.updateMeters(pos_ppl_meters, res.pos_ppl)
                    end
                else
                    local res = optHiddens(curr_hid_inputs,
                                      curr_hid_targets, curr_grad_hids,
                                      curr_duals, curr_inputs, curr_targets)
                    h_xent = h_xent + res['hxent']
                    h_l2   = h_l2 + res['hl2']
                end

                currloss = h_xent + h_l2
                opt_steps = opt_steps + 1
                -- scale and update gradients
                grad_hids:mul(1/blocks_in_update)
                updateHiddens(hids, grad_hids, h_opt_state)

                if config.alm then -- also update parameters
                    scaleGradients(grad_params, 1/blocks_in_update)
                    updateParams(params, grad_params, p_opt_state)

                    if config.gd and currloss > prevloss then
                        p_opt_state.lr = p_opt_state.lr/2
                        h_opt_state.lr = h_opt_state.lr/2
                    end

                    if torch.sqrt(grad_hids:norm(2)^2
                      + gradsNorm(grad_params)^2) < config.alm_grad_tol then
                          break
                    end

                    if config.gd and p_opt_state.lr < 1e-5
                        and h_opt_state.lr < 1e-5 then break end
                elseif grad_hids:norm(2) < config.h_tol then -- admm case
                        break
                end
                prevloss = currloss
            end

            if config.alm and config.mb_init then
                last_output:copy(recurrentLayer.output[-1][-1])
            end

            --if minibatching and pre-initing, do u's before params
            if epoch >= config.u_startupdate and config.mb_init then
                grad_duals:zero()
                local res = optDuals(curr_hid_inputs, curr_hid_targets,
                                     curr_duals, curr_grad_duals,
                                     curr_inputs)
                u_l2 = u_l2 + res['ul2']
                -- scale the gradients and update
                grad_duals:mul(1/blocks_in_update)
                updateDuals(duals, grad_duals, u_opt_state)
            end

            -- move onto params if alternating
            if not config.alm then
                local prevloss, currloss = math.huge, math.huge
                for step = 1, config.param_maxsteps do
                    trainGraph:zeroGradParameters()

                    local res = optParams(curr_hid_inputs, curr_hid_targets,
                                          curr_duals, curr_inputs,
                                          curr_targets, params, grad_params)
                    if config.get_positional_ppl then
                        lutils.updateMeters(pos_ppl_meters, res.ppos_ppl)
                    end
                    p_xent = p_xent + res['pxent']
                    p_l2   = p_l2 + res['pl2']

                    currloss = p_xent + p_l2
                    -- scale and update gradients
                    scaleGradients(grad_params, 1/blocks_in_update)
                    updateParams(params, grad_params, p_opt_state)

                    if config.gd and currloss > prevloss then
                        p_opt_state.lr = p_opt_state.lr/2
                    end
                    if config.gd and p_opt_state.lr < 1e-5 then break end
                    prevloss = currloss
                end
                -- copy last hidden state of final block in mini-batch
                -- to initialize first hidden state of next block
                if config.mb_init then
                    last_output:copy(recurrentLayer.output[-1][-1])
                end
            end
            start_batch = stop_batch + 1
            stop_batch = math.min(start_batch, nbatches)
        end -- end for uu

        u_l2 = u_l2 / nblocks
        meters['ul2']:add(u_l2)

        h_xent = h_xent / nblocks
        h_l2   = h_l2 / nblocks

        p_xent = p_xent / nblocks
        p_l2   = p_l2 / nblocks
        -- update meters
        if config.alm then
            meters['xent']:add(h_xent)
            meters['l2']:add(h_l2)
            meters['step']:add(opt_steps)
        else
            meters['hxent']:add(h_xent)
            meters['hl2']:add(h_l2)
            meters['hstep']:add(opt_steps)
            meters['pxent']:add(p_xent)
            meters['pl2']:add(p_l2)
        end
        meters['lag']:add(h_xent + h_l2)

        -- display the meters
        lutils.displayMeters(meters)
        if config.get_positional_ppl then
            print("positional ppl")
            lutils.displayMeters(pos_ppl_meters)
        end

        -- evaluate model
        local trppl = evaluate(smalltrain)
        local vlppl = evaluate(valid)
        io.write(string.format('%10s -- %3.4f\n',   'train ppl', trppl))
        io.write(string.format('%10s -- %3.4f\n\n', 'valid ppl', vlppl))

        if trppl ~= trppl or trppl > config.max_ppl or vlppl ~= vlppl then
            print("-- train perplexity not looking good; bailing")
            trppl = config.max_ppl
            vlppl = config.max_ppl
        end

        if vlppl < best_vlppl then
            best_vlppl = vlppl
            best_trppl = trppl
            last_good_epoch = epoch
            if config.save_params then
                print("saving params!")
                torch.save("best-lm-lut.t7." .. timestr, lut)
                torch.save("best-lm-stepnet.t7." .. timestr, step_net)
                torch.save("best-lm-decoder.t7." .. timestr, decoder)
            end
        end

        if epoch - last_good_epoch > config.tol then
            io.write(string.format("-- Early stopping. Tol reached\n",
                                   config.tol))
           if trppl ~= trppl then
               trppl = -1
           end
           if vlppl ~= vlppl then
               vlppl = -1
           end
           break
        end

        if config.optim == 'gd' then
            if (h_opt_state.lr < 1e-5 or
                    p_opt_state.lr < 1e-5 or
                u_opt_state.lr < 1e-5) then
                print("-- Early stopping since learning rates are 0")
                break
            end
        end

    end
end

print(config)
do_training(train, valid, smalltrain)

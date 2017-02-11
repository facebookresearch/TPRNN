--[[
Copyright (c) 2017-present, Facebook, Inc.
All rights reserved.

This source code is licensed under the CC-NC style license found in the
LICENSE file in the root directory of this source tree.
--]]

require 'nn'
require 'nngraph'
dofile 'scLSTM.lua'

local mdls = {}

-- we'll use the convention that the first input is the input and the second
-- is the previous state, as in rnnlib
function mdls.make_elman_net(opt)
    local model = nn.Sequential()
    :add(nn.ParallelTable()
         :add(nn.Linear(opt.nhid, opt.nhid, false))--true))
         :add(nn.Linear(opt.nhid, opt.nhid, false)))
    :add(nn.CAddTable())
    :add(nn.Sigmoid())
    return model
end

function mdls.make_mult_elman_net(opt)
    local model = nn.Sequential()
        :add(nn.ParallelTable()
            :add(nn.Linear(opt.nhid, opt.nhid, true))
            :add(nn.Linear(opt.nhid, opt.nhid, true)))
        :add(nn.CMulTable())
        --:add(nn.Add(opt.nhid)) -- seems redundant
    return model
end

-- Eqn (4) in http://arxiv.org/pdf/1606.06630.pdf
function mdls.make_gateder_mult_elman_net(opt)
    local x = nn.Identity()()
    local prevh = nn.Identity()()
    local Ax = nn.Linear(opt.nhid, opt.nhid, false)(x)
    local Cprevh = nn.Linear(opt.nhid, opt.nhid, false)(prevh)
    local gated_prod = nn.CMul(opt.nhid)(nn.CMulTable()({Ax, Cprevh}))
    local mult_C = nn.CMul(opt.nhid)(Cprevh)
    local mult_A = nn.CMul(opt.nhid)(Ax)
    local outp = nn.Add(opt.nhid)(nn.CAddTable(){gated_prod, mult_C, mult_A})
    local model = nn.gModule({x, prevh}, {outp})
    return model
end

-- implements h_t = X_t h_{t-1}, where X_t is a dxd matrix
-- assumes external lut
function mdls.make_unfactored_mult_rnn(opt)
    local model = nn.Sequential()
        :add(nn.ParallelTable()
            :add(nn.View(opt.nhid, opt.nhid))
            :add(nn.Identity()))
        :add(nn.MV())
        --:add(nn.Add(opt.nhid)) -- no bias!
    return model
end

function mdls.make_lstm_net(opt)
    -- adapted from rnn-lib-v2
    local function _makeLinear(input, insize, outsize, initv)
        local node = nn.Linear(insize, outsize, false)
        initv      = initv or 0.1
        -- init the weights:
        node.weight:uniform(-initv, initv)
        return node(input)
    end
    local input = nn.Identity()()
    local prevcandh = nn.Identity()()
    local prevc = nn.Narrow(2, 1, opt.nhid)(prevcandh)
    local prevh = nn.Narrow(2, opt.nhid+1, opt.nhid)(prevcandh)

    -- the four gates are computed simulatenously
    local nhid  = opt.nhid
    local i2h   = _makeLinear(input, nhid, 4 * nhid, opt.init_range)
    local h2h   = _makeLinear(prevh, nhid, 4 * nhid, opt.init_range)
    -- the gates are separated
    local gates = nn.CAddTable()({i2h, h2h})
    gates       = nn.SplitTable(2)(nn.Reshape(4, nhid)(gates))
    -- apply nonlinearities:
    local ingate     = nn.Sigmoid()(nn.SelectTable(1)(gates))
    local cellgate   = nn.Tanh()(   nn.SelectTable(2)(gates))
    local forgetgate = nn.Sigmoid()(nn.SelectTable(3)(gates))
    local outgate    = nn.Sigmoid()(nn.SelectTable(4)(gates))
    -- c_{t+1} = forgetgate * c_t + inputgate * f(h_{t+1}, i_{t+1})
    local nextc = nn.CAddTable()({
        nn.CMulTable()({forgetgate, prevc}),
        nn.CMulTable()({ingate, cellgate})
    })
    -- h_{t+1} = outgate * c_{t+1}
    local nexth = nn.CMulTable()({outgate,nn.Tanh()(nextc)})
    local nextcandh = nn.JoinTable(2)({nextc, nexth})
    local model = nn.gModule({input, prevcandh}, {nextcandh})
    return model
end

function mdls.make_lstm_decoder(ntypes, opt)
    local dec = nn.Sequential()
        :add(nn.Narrow(2, opt.nhid+1, opt.nhid))
        :add(nn.Linear(opt.nhid, ntypes+1))
        :add(nn.LogSoftMax())
    return dec
end

function mdls.make_cudnn_rnntanh(nhid, ntypes)
    -- input lookup table
    local lut = nn.LookupTable(ntypes, nhid)
    -- recurrent layer
    local rec = cudnn.RNNTanh(nhid, nhid, 1, false)
    -- decoder
    local dec = nn.Sequential()
    dec:add(nn.View(-1, nhid))
    dec:add(nn.Linear(nhid, ntypes))
    dec:add(nn.LogSoftMax())

    -- full network
    local gnet = nn.Sequential()
    gnet:add(lut)
    gnet:add(rec)

    -- make the network graph
    local input = nn.Identity()()
    local xs, us = input:split(2)
    local hids = gnet(xs)
    local preds = nn.Select(1, -1)(hids)
    local out1 = dec(hids)
    local out2 = nn.CAddTable()({preds, us})
    local traingraph = nn.gModule({input}, {out1, out2}):cuda()

    return lut, rec, dec, gnet, traingraph
end


function mdls.make_cudnn_lstm(nhid, ntypes)
    -- input lookup table
    local lut = nn.LookupTable(ntypes, nhid)
    -- recurrent layer
    local rec = cudnn.scLSTM(nhid, nhid, 1) -- , false, false, 0, true)
    -- decoder
    local dec = nn.Sequential()
    dec:add(nn.View(-1, nhid))
    dec:add(nn.Linear(nhid, ntypes))
    dec:add(nn.LogSoftMax())

    -- full network
    local gnet = nn.Sequential()
    gnet:add(lut)
    gnet:add(rec)

    -- make the network graph
    local input = nn.Identity()()
    local xs, us = input:split(2)
    local hids, cell = gnet(xs):split(2)
    local preds_hid = nn.Select(1, -1)(hids)
    local preds_cel = nn.Select(1, 1)(cell)
    --local preds_hid = last_hid
    --local preds_cel = last_cel
    local preds = nn.JoinTable(2)({preds_hid, preds_cel})
    local out1 = dec(hids)
    local out2 = nn.CAddTable()({preds, us})
    local traingraph = nn.gModule({input}, {out1, out2}):cuda()

    return lut, rec, dec, gnet, traingraph
end

function mdls.make_cudnn_gru(nhid, ntypes)
    -- input lookup table
    local lut = nn.LookupTable(ntypes, nhid)
    -- recurrent layer
    local rec = cudnn.GRU(nhid, nhid, 1, false)
    -- decoder
    local dec = nn.Sequential()
    dec:add(nn.View(-1, nhid))
    dec:add(nn.Linear(nhid, ntypes))
    dec:add(nn.LogSoftMax())

    -- full network
    local gnet = nn.Sequential()
    gnet:add(lut)
    gnet:add(rec)

    -- make the network graph
    local input = nn.Identity()()
    local xs, us = input:split(2)
    local hids = gnet(xs)
    local preds = nn.Select(1, -1)(hids)
    local out1 = dec(hids)
    local out2 = nn.CAddTable()({preds, us})
    local traingraph = nn.gModule({input}, {out1, out2}):cuda()

    return lut, rec, dec, gnet, traingraph
end


function mdls.make_cudnn_rnntanh_penalty(nhid, ntypes)
    -- input lookup table
    local lut = nn.LookupTable(ntypes, nhid)
    -- recurrent layer
    local rec = cudnn.RNNTanh(nhid, nhid, 1, false)
    -- decoder
    local dec = nn.Sequential()
    dec:add(nn.View(-1, nhid))
    dec:add(nn.Linear(nhid, ntypes))
    dec:add(nn.LogSoftMax())
    -- full network
    local gnet = nn.Sequential()
    gnet:add(lut)
    gnet:add(rec)
    -- make the network graph
    local input = nn.Identity()()
    local hids  = gnet(input)
    local out   = dec(hids)
    local preds = nn.Select(1, -1)(hids)
    local traingraph = nn.gModule({input}, {out, preds}):cuda()
    return lut, rec, dec, gnet, traingraph
end

function mdls.make_cudnn_lstm_penalty(nhid, ntypes)
    local lut = nn.LookupTable(ntypes, nhid)
    local rec = cudnn.scLSTM(nhid, nhid, 1)
    local dec = nn.Sequential()
    dec:add(nn.View(-1, nhid))
    dec:add(nn.Linear(nhid, ntypes))
    dec:add(nn.LogSoftMax())
    -- full network
    local gnet = nn.Sequential()
    gnet:add(lut)
    gnet:add(rec)
    -- make the network graph
    local input = nn.Identity()()
    local hids, cell = gnet(input):split(2)
    local preds_hid  = nn.Select(1, -1)(hids)
    local preds_cel  = nn.Select(1, 1)(cell)
    local preds      = nn.JoinTable(2)({preds_hid, preds_cel})
    local out        = dec(hids)
    local traingraph = nn.gModule({input}, {out, preds}):cuda()

    return lut, rec, dec, gnet, traingraph
end

function mdls.make_cudnn_gru_penalty(nhid, ntypes)
    local lut = nn.LookupTable(ntypes, nhid)
    local rec = cudnn.GRU(nhid, nhid, 1, false)
    local dec = nn.Sequential()
    dec:add(nn.View(-1, nhid))
    dec:add(nn.Linear(nhid, ntypes))
    dec:add(nn.LogSoftMax())
    -- full network
    local gnet = nn.Sequential()
    gnet:add(lut)
    gnet:add(rec)
    -- make the network graph
    local input = nn.Identity()()
    local hids  = gnet(input)
    local preds = nn.Select(1, -1)(hids)
    local out   = dec(hids)
    local traingraph = nn.gModule({input}, {out, preds}):cuda()

    return lut, rec, dec, gnet, traingraph
end



return mdls

--[[
Copyright (c) 2017-present, Facebook, Inc.
All rights reserved.

This source code is licensed under the CC-NC style license found in the
LICENSE file in the root directory of this source tree.
--]]

local utils = {}


utils.printf = function(...) print(string.format(...)) end


-- create the directory where check points will be saved
utils.setSaveDir = function(opt, suffix)
    local mdir =
        'nhid=' .. opt.nhid ..
        '_nlayers=' .. opt.nlayer ..
        '_bptt=' .. opt.bptt ..
        '_batchsize=' .. opt.batchsize ..
        '_updatefreq=' .. opt.update_freq ..
        '_lr=' .. opt.lr ..
        '_lrshrink=' .. opt.lrshrink ..
        '_clip=' .. opt.clip ..
        '_optim=' .. opt.optim
    if suffix ~= nil and suffix ~= '' then
        mdir = mdir .. '_' .. suffix
    end
    local basedir = '/tmp/rnnlm_checkpoints/' .. opt.dset .. '/' .. opt.model
    return paths.concat(basedir, mdir)
end


utils.trainLogStr = function(epch, nex, lr, time, commtime, loss, dset)
    local loss2b = loss/math.log(2)
    local str = string.format(
        '| epoch %03d | %05d samples | lr %02.6f | ms/epoch %3d |'
            .. ' comm ms/epoch %3d | train loss %5.2f | train ppl %8.2f',
        epch, nex, lr, time * 1000, commtime * 1000, loss2b, math.pow(2, loss2b))
    return str
end


utils.validLogStr = function(epch, loss, dset)
    local loss2b = loss/math.log(2)
    local str = string.format(
            '| epoch %03d | valid loss %5.2f | valid ppl %8.2f',
            epch, loss2b, math.pow(2, loss2b))
    return str
end


utils.setUpdateParametersMethod = function(optim)
    if optim == 'sgd' then
        function nn.Module:updateParameters(learningRate)
            local params, gradParams = self:parameters()
            if params then
                for i=1, #params do
                    params[i]:add(-learningRate, gradParams[i])
                end
            end
        end

    elseif optim == 'adagrad' then
        function nn.Module:updateParameters(learningRate)
            local allParams, allGradParams = self:parameters()
            if allParams then
                self.state = self.state or {}
                self.lr = self.lr or {}
                self.lrd = self.lrd or {}
                for pidx = 1, #allParams do
                    params = allParams[pidx]
                    gradParams = allGradParams[pidx]
                    self.state[pidx] = self.state[pidx] or {}
                    self.lr[pidx] = self.lr[pidx] or learningRate
                    self.lrd[pidx] = self.lrd[pidx] or 0
                    self.state[pidx].evalCounter =
                        self.state[pidx].evalCounter or 0
                    local nevals = self.state[pidx].evalCounter

                    -- learning rate decay (annealing)
                    local clr = self.lr[pidx] / (1 + nevals * self.lrd[pidx])

                    -- parameter update with single or individual lr
                    if not self.state[pidx].paramVariance then
                        self.state[pidx].paramVariance =
                            torch.Tensor():typeAs(
                                gradParams):resizeAs(gradParams):zero()
                        self.state[pidx].paramStd =
                            torch.Tensor():typeAs(
                                gradParams):resizeAs(gradParams)
                    end
                    self.state[pidx].paramVariance:addcmul(
                        1, gradParams, gradParams)
                    self.state[pidx].paramStd:resizeAs(
                        self.state[pidx].paramVariance):copy(
                        self.state[pidx].paramVariance):sqrt()
                    params:addcdiv(-clr, gradParams,
                                   self.state[pidx].paramStd:add(1e-10))
                    self.state[pidx].evalCounter =
                        self.state[pidx].evalCounter + 1
                end
            end
        end

    elseif optim == 'rmsprop' then
        function nn.Module:updateParameters(learningRate)
            local allParams, allGradParams = self:parameters()
            if allParams then
                self.state = self.state or {}
                self.lr    = self.lr or learningRate
                self.rho   = self.rho or 0.9
                self.eps   = self.eps or 1e-3
                for pidx = 1, #allParams do
                    local params = allParams[pidx]
                    local gradParams = allGradParams[pidx]
                    self.state[pidx] = self.state[pidx] or {}
                    -- parameter update
                    if not self.state[pidx].paramVariance then
                        self.state[pidx].paramVariance =
                            torch.Tensor():typeAs(gradParams):resizeAs(
                                gradParams):zero()
                        self.state[pidx].paramStd =
                            torch.Tensor():typeAs(gradParams):resizeAs(
                                gradParams):zero()
                        self.state[pidx].delta =
                            torch.Tensor():typeAs(gradParams):resizeAs(
                                gradParams):zero()
                    end
                    local paramVariance = self.state[pidx].paramVariance
                    local paramStd      = self.state[pidx].paramStd
                    local delta         = self.state[pidx].delta

                    paramVariance:mul(self.rho):addcmul(
                        1 - self.rho, gradParams, gradParams)
                    paramStd:resizeAs(paramVariance):copy(
                        paramVariance):add(self.eps):sqrt()
                    delta:resizeAs(paramVariance):fill(self.lr):cdiv(
                        paramStd):cmul(gradParams)
                    params:add(-1, delta)
                end
            end
        end

    elseif optim == 'adadelta' then
        function nn.Module:updateParameters(learningRate)
            local allParams, allGradParams = self:parameters()
            if allParams then
                self.state = self.state or {}
                self.rho   = self.rho or 0.9
                self.eps   = self.eps or 1e-3
                for pidx = 1, #allParams do
                    local params = allParams[pidx]
                    local gradParams = allGradParams[pidx]
                    self.state[pidx] = self.state[pidx] or {}
                    -- parameter update
                    if not self.state[pidx].paramVariance then
                        self.state[pidx].paramVariance =
                            torch.Tensor():typeAs(gradParams):resizeAs(
                                gradParams):zero()
                        self.state[pidx].paramStd = torch.Tensor():typeAs(
                            gradParams):resizeAs(gradParams):zero()
                        self.state[pidx].delta = torch.Tensor():typeAs(
                            gradParams):resizeAs(gradParams):zero()
                        self.state[pidx].accDelta = torch.Tensor():typeAs(
                            gradParams):resizeAs(gradParams):zero()
                    end
                    local paramVariance = self.state[pidx].paramVariance
                    local paramStd      = self.state[pidx].paramStd
                    local delta         = self.state[pidx].delta
                    local accDelta      = self.state[pidx].accDelta

                    paramVariance:mul(self.rho):addcmul(
                        1 - self.rho, gradParams, gradParams)
                    paramStd:resizeAs(paramVariance):copy(
                        paramVariance):add(self.eps):sqrt()
                    delta:resizeAs(paramVariance):copy(accDelta):add(
                        self.eps):sqrt():cdiv(paramStd):cmul(gradParams)
                    params:add(-1, delta)
                    accDelta:mul(self.rho):addcmul(1 - self.rho, delta, delta)
                end
            end
        end

    elseif optim == 'adam' then
        function nn.Module:updateParameters(learningRate)
            local allParams, allGradParams = self:parameters()
            if allParams then
                self.state = self.state or {}
                self.lr    = self.lr or learningRate
                self.beta1 = self.beta1 or 0.9
                self.beta2 = self.beta2 or 0.999
                self.eps   = self.eps or 1e-8
                for pidx = 1, #allParams do
                    local params = allParams[pidx]
                    local gradParams = allGradParams[pidx]
                    self.state[pidx] = self.state[pidx] or {}
                    -- parameter update
                    if not self.state[pidx].t then
                        self.state[pidx].t = 1
                        self.state[pidx].m = torch.Tensor():typeAs(
                            gradParams):resizeAs(gradParams):zero()
                        self.state[pidx].v = torch.Tensor():typeAs(
                            gradParams):resizeAs(gradParams):zero()
                        self.state[pidx].denom = torch.Tensor():typeAs(
                            gradParams):resizeAs(gradParams):zero()
                    else
                        self.state[pidx].t = self.state[pidx].t + 1
                    end
                    local t     = self.state[pidx].t
                    local m     = self.state[pidx].m
                    local v     = self.state[pidx].v
                    local denom = self.state[pidx].denom

                    m:mul(self.beta1):add(1 - self.beta1, gradParams)
                    v:mul(self.beta2):addcmul(
                        1 - self.beta2, gradParams, gradParams)
                    denom:copy(v):sqrt():add(self.eps)

                    local biasCorrection1 = 1 - self.beta1^t
                    local biasCorrection2 = 1 - self.beta2^t
                    local stepSize =
                        self.lr * math.sqrt(biasCorrection2)/biasCorrection1
                    -- update parameters
                    params:addcdiv(-stepSize, m, denom)
                end
            end
        end
    end
end


function utils.clipGradients(grads, norm)
    local totalnorm = 0
    for mm = 1, #grads do
        local modulenorm = grads[mm]:norm()
        totalnorm = totalnorm + modulenorm * modulenorm
    end
    totalnorm = math.sqrt(totalnorm)
    if totalnorm > norm then
        local coeff = norm / math.max(totalnorm, 1e-6)
        for mm = 1, #grads do
            grads[mm]:mul(coeff)
        end
    end
end


function utils.nag_step(x, dfdx, state)
    if torch.type(x) == 'table' then -- run for each thing in the table
        for i, v in pairs(x) do
            if not state[i] then
                state[i] = {}
            end
            state[i].lr = state.lr
            utils.tensor_nag_step(v, dfdx[i], state[i])
        end
    else
        utils.tensor_nag_step(x, dfdx, state)
    end
end


function utils.adagrad_step(allParams, allGradParams, learningRate, state)
    local state = state or {}
    state.lr = state.lr or {}
    state.lrd = state.lrd or {}

    for pidx = 1, #allParams do
        local params = allParams[pidx]
        local gradParams = allGradParams[pidx]
        state[pidx]     = state[pidx]     or {}
        state.lr[pidx]  = state.lr[pidx]  or learningRate
        state.lrd[pidx] = state.lrd[pidx] or 0
        state[pidx].evalCounter = state[pidx].evalCounter or 0
        local nevals = state[pidx].evalCounter

        -- learning rate decay (annealing)
        local clr = state.lr[pidx] / (1 + nevals * state.lrd[pidx])

        -- parameter update with single or individual learning rates
        if not state[pidx].paramVariance then
            state[pidx].paramVariance =
                torch.Tensor():typeAs(gradParams):resizeAs(gradParams):zero()
            state[pidx].paramStd =
                torch.Tensor():typeAs(gradParams):resizeAs(gradParams)
        end
        state[pidx].paramVariance:addcmul(1, gradParams, gradParams)
        state[pidx].paramStd:resizeAs(
            state[pidx].paramVariance):copy(
            state[pidx].paramVariance):sqrt()
        params:addcdiv(-clr, gradParams, state[pidx].paramStd:add(1e-5))
        state[pidx].evalCounter = state[pidx].evalCounter + 1
    end
    return state
end


function utils.resetMeters(mtrs)
    for i, v in pairs(mtrs) do
        v:reset()
    end
end


function utils.displayMeters(mtrs)
    for i, v in pairs(mtrs) do
        local mean, std = v:value()
        io.write(string.format('%10s -- %2.5f\n', i, mean))
    end
end


function utils.updateMeters(mtrs, val_tbl)
    for k, v in pairs(val_tbl) do
        if mtrs[k] then
            mtrs[k]:add(v)
        end
    end
end





return utils

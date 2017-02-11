--[[
Copyright (c) 2017-present, Facebook, Inc.
All rights reserved.

This source code is licensed under the CC-NC style license found in the
LICENSE file in the root directory of this source tree.
--]]

require 'cudnn'

local scLSTM, parent = torch.class('cudnn.scLSTM', 'cudnn.LSTM')

function scLSTM:__init(inputSize, hiddenSize, numLayers, batchFirst, dropout, rememberStates)
    parent.__init(self, inputSize, hiddenSize, numLayers, batchFirst, dropout, rememberStates)
end

function scLSTM:updateOutput(input)
    local hiddenOutput = parent.updateOutput(self, input)
    local cellOutput   = self.cellOutput
    return {hiddenOutput, cellOutput}
end

-- gradOutput is a table: {gradHiddenOutput, gradCellOutput}
function scLSTM:updateGradInput(input, gradOutput)
    local gradHiddenOutput = gradOutput[1]
    local gradCellOutput   = gradOutput[2]

    self.gradCellOutput = gradCellOutput
    self.gradInput = parent.updateGradInput(self, input, gradHiddenOutput)
    return self.gradInput
end

-- gradOutput is a table: {gradHiddenOutput, gradCellOutput}
function scLSTM:accGradParameters(input, gradOutput)
    local gradHiddenOutput = gradOutput[1]
    local gradCellOutput   = gradOutput[2]
    self.gradCellOutput = gradCellOutput
    parent.accGradParameters(self, input, gradHiddenOutput)
end

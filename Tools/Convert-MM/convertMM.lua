--[[
    Author: Aysegul Dundar (adundar@purdue.edu)
    This file converts GPU trained ConvolutionMM module to Spatial Convolution
    Because ConvolutionMM for GPU has the ability of having stride more than 1
    whereas ConvolutionMM for CPU has not.
]]

require 'cutorch'
require 'cunn'
require 'nnx'

local network = torch.load('model.net'):float()

-- stride pre-defined
local dW = 4
local dH = 4

local m_tmp = network.modules[1].modules[1]
local conv_tmp = nn.SpatialConvolution(m_tmp.nInputPlane, m_tmp.nOutputPlane, m_tmp.kW, m_tmp.kH, dW, dH)
conv_tmp.weight = m_tmp.weight:reshape(m_tmp.nOutputPlane, m_tmp.nInputPlane, m_tmp.kW, m_tmp.kH)
conv_tmp.bias   = m_tmp.bias

network.modules[1].modules[1] = conv_tmp

torch.save('model.net', network)

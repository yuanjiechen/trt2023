/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cuda.h>


#include <cstring>
#include <vector>

#include "NvInfer.h"
#include "common/bertCommon.h"
#include "common/serialize.hpp"
#include "geluPlugin.h"

using namespace nvinfer1;
using namespace nvinfer1::plugin;
using namespace nvinfer1::plugin::bert;

namespace
{
char const* const kGELU_PLUGIN_VERSION{"1"};
char const* const kGELU_PLUGIN_NAME{"CustomGeluPlugin"};
} // namespace

// Static class fields initialization
PluginFieldCollection GeluPluginCreator::mFC{};
std::vector<PluginField> GeluPluginCreator::mPluginAttributes;

REGISTER_TENSORRT_PLUGIN(GeluPluginCreator);

GeluPlugin::GeluPlugin(const std::string name)
    : mLayerName(name)

{
    // mHasBias = (bias.values != nullptr);
    // if (mHasBias)
    // {
    //     void* cudaMem{nullptr};
    //     PLUGIN_CUASSERT(cudaMalloc(&cudaMem, getWeightsSize(bias, mType)));
    //     PLUGIN_CUASSERT(cudaMemcpy(cudaMem, bias.values, getWeightsSize(bias, mType), cudaMemcpyHostToDevice));
    //     make_cuda_shared(mBiasDev, cudaMem);
    // }
}

GeluPlugin::GeluPlugin(const std::string name, void const* data, size_t length)
    : mLayerName(name)
{
    gLogVerbose << "GeluPlugin deserialize\n";
    deserialize_value(&data, &length, &mType);
    deserialize_value(&data, &length, &mLd);
    deserialize_value(&data, &length, &mHasBias);

    if (mHasBias)
    {
        PLUGIN_VALIDATE(mLd > 0);
        char const* d = static_cast<char const*>(data);
        make_cuda_shared(mBiasDev, deserToDev<char>(d, mLd * getElementSize(mType)));
    }
}
// IPluginV2DynamicExt Methods
nvinfer1::IPluginV2DynamicExt* GeluPlugin::clone() const noexcept
{
    try
    {
        gLogVerbose << "GeluPlugin clone\n";
        auto* plugin = new GeluPlugin(*this);
        plugin->setPluginNamespace(mNamespace.c_str());
        return plugin;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

nvinfer1::DimsExprs GeluPlugin::getOutputDimensions(int32_t outputIndex, nvinfer1::DimsExprs const* inputs,
    int32_t nbInputs, nvinfer1::IExprBuilder& exprBuilder) noexcept
{   
    // std::cout << inputs[0].d[0].getConstantValue() << inputs[0].d[1].getConstantValue() << inputs[0].d[2].getConstantValue() << "\n";

    return inputs[0];
    // try
    // {
    //     PLUGIN_VALIDATE(inputs != nullptr);
    //     PLUGIN_VALIDATE(nbInputs == 1);
    //     PLUGIN_VALIDATE(outputIndex == 0);
    //     return inputs[0];
    // }
    // catch (std::exception const& e)
    // {
    //     caughtError(e);
    // }
    // return DimsExprs{};
}

bool GeluPlugin::supportsFormatCombination(
    int32_t pos, nvinfer1::PluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept
{
    // try
    // {
    //     PLUGIN_VALIDATE(inOut != nullptr);
    //     PLUGIN_VALIDATE(nbInputs == 1);
    //     PLUGIN_VALIDATE(nbOutputs == 1);
    //     PLUGIN_VALIDATE(pos >= 0);
    //     PLUGIN_VALIDATE(pos < nbInputs + nbOutputs);
    // }
    // catch (std::exception const& e)
    // {
    //     caughtError(e);
    //     return false;
    // }

    // PluginTensorDesc const& input = inOut[0];
    if (pos == 0)
    {   //std::cout << int32_t(inOut[0].format) << "\n";
        return (inOut[0].type == DataType::kFLOAT || inOut[0].type == DataType::kHALF) && (inOut[0].format == TensorFormat::kLINEAR || inOut[0].format == TensorFormat::kCHW2 || inOut[0].format == TensorFormat::kHWC8 || inOut[0].format == TensorFormat::kCHW4 || inOut[0].format == TensorFormat::kCHW32 || inOut[0].format == TensorFormat::kHWC16);
    }
    if (pos == 1 || pos == 2)
    {
        return (inOut[pos].type == inOut[0].type) && (inOut[pos].format == inOut[0].format);
    }
    return false;
}

void GeluPlugin::configurePlugin(nvinfer1::DynamicPluginTensorDesc const* in, int32_t nbInputs,
    nvinfer1::DynamicPluginTensorDesc const* out, int32_t nbOutputs) noexcept
{
    gLogVerbose << "GeluPlugin configurePlugin\n";

    // try
    // {
    //     PLUGIN_VALIDATE(in != nullptr);
    //     PLUGIN_VALIDATE(nbInputs == 1);
    //     PLUGIN_VALIDATE(mType == in[0].desc.type);
    // }
    // catch (std::exception const& e)
    // {
    //     caughtError(e);
    // }
}

size_t GeluPlugin::getWorkspaceSize(nvinfer1::PluginTensorDesc const* inputs, int32_t nbInputs,
    nvinfer1::PluginTensorDesc const* outputs, int32_t nbOutputs) const noexcept
{
    return 0;
}

template <typename TDataType>
int32_t GeluPlugin::enqueueTyped(
    void const* input_, void const* x_, void* output_, int32_t const inputVolume, cudaStream_t stream) noexcept
{
    TDataType const* input = static_cast<TDataType const*>(input_);
    TDataType const* x = static_cast<TDataType const*>(x_);
    TDataType* output = static_cast<TDataType*>(output_);
    // int32_t const cols = inputVolume / mLd;
    // int32_t const rows = mLd;

    // if (mHasBias)
    // {
    //     TDataType const* bias = static_cast<TDataType*>(mBiasDev.get());
    //     return computeGeluBias(output, input, bias, rows, cols, stream);
    // }
    // else
    // {
    return computeGelu(stream, inputVolume, input, x, output);
    // }
}

int32_t GeluPlugin::enqueue(nvinfer1::PluginTensorDesc const* inputDesc,
    nvinfer1::PluginTensorDesc const* outputDesc, void const* const* inputs, void* const* outputs, void* workspace,
    cudaStream_t stream) noexcept
{
    try
    {
        PLUGIN_VALIDATE(inputDesc != nullptr);
        PLUGIN_VALIDATE(inputs != nullptr);
        PLUGIN_VALIDATE(outputs != nullptr);
    }
    catch (std::exception const& e)
    {
        caughtError(e);
        return STATUS_FAILURE;
    }

    int32_t const inputVolume = volume(inputDesc[0].dims);

    // Our plugin outputs only one tensor.
    // Launch CUDA kernel wrapper and save its return value.
    switch (inputDesc[0].type)
    {
    case DataType::kFLOAT: return enqueueTyped<float>(inputs[0], inputs[1], outputs[0], inputVolume, stream);
    case DataType::kHALF: return enqueueTyped<half>(inputs[0], inputs[1], outputs[0], inputVolume, stream);
    default: return STATUS_FAILURE;
    }
}

// IPluginV2Ext Methods
nvinfer1::DataType GeluPlugin::getOutputDataType(
    int32_t index, nvinfer1::DataType const* inputTypes, int32_t nbInputs) const noexcept
{
    try
    {
        // PLUGIN_VALIDATE(index == 0);
        PLUGIN_VALIDATE(inputTypes != nullptr);
        PLUGIN_VALIDATE(inputTypes[0] == DataType::kFLOAT || inputTypes[0] == DataType::kHALF);
        return inputTypes[0];
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return DataType{};
}

// IPluginV2 Methods

char const* GeluPlugin::getPluginType() const noexcept
{
    return kGELU_PLUGIN_NAME;
}

char const* GeluPlugin::getPluginVersion() const noexcept
{
    return kGELU_PLUGIN_VERSION;
}

int32_t GeluPlugin::getNbOutputs() const noexcept
{
    return 1;
}

int32_t GeluPlugin::initialize() noexcept
{
    gLogVerbose << "GeluPlugin initalize\n";
    return 0;
}

void GeluPlugin::terminate() noexcept
{
    gLogVerbose << "GeluPlugin terminate\n";
}

size_t GeluPlugin::getSerializationSize() const noexcept
{
    const size_t wordSize = getElementSize(mType);
    // const size_t biasSize = mHasBias ? mLd * wordSize : 0;
    return sizeof(mType) + sizeof(mHasBias) + sizeof(mLd); // + biasSize;
}

void GeluPlugin::serialize(void* buffer) const noexcept
{
    // serialize_value(&buffer, mType);
    // serialize_value(&buffer, mLd);
    // serialize_value(&buffer, mHasBias);
    // if (mHasBias)
    // {
    //     PLUGIN_ASSERT(mLd > 0);
    //     char* d = static_cast<char*>(buffer);
    //     serFromDev(d, static_cast<char*>(mBiasDev.get()), mLd * getElementSize(mType));
    // }
}

void GeluPlugin::destroy() noexcept
{
    gLogVerbose << "GeluPlugin destroy\n";
    // This gets called when the network containing plugin is destroyed
    mBiasDev.reset();
    delete this;
}

void GeluPlugin::setPluginNamespace(char const* libNamespace) noexcept
{
    try
    {
        PLUGIN_VALIDATE(libNamespace != nullptr);
        mNamespace = libNamespace;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
}

char const* GeluPlugin::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

///////////////

GeluPluginCreator::GeluPluginCreator()
{
    // mPluginAttributes.emplace_back(PluginField("type_id", nullptr, PluginFieldType::kINT32, 1));
    // mPluginAttributes.emplace_back(PluginField("bias", nullptr, PluginFieldType::kFLOAT32, 1));

    // Fill PluginFieldCollection with PluginField arguments metadata
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = nullptr; //mPluginAttributes.data();
}

char const* GeluPluginCreator::getPluginName() const noexcept
{
    return kGELU_PLUGIN_NAME;
}

char const* GeluPluginCreator::getPluginVersion() const noexcept
{
    return kGELU_PLUGIN_VERSION;
}

PluginFieldCollection const* GeluPluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

IPluginV2* GeluPluginCreator::createPlugin(char const* name, PluginFieldCollection const* fc) noexcept
{
    try
    {
        gLogVerbose << "GeluPluginCreator createPlugin\n";
        // PLUGIN_VALIDATE(fc != nullptr);
        auto* plugin = new GeluPlugin(name);
        plugin->setPluginNamespace(mNamespace.c_str());
        // Weights bias{DataType::kFLOAT, nullptr, 0};
        // int32_t typeId = -1;
        // plugin::validateRequiredAttributesExist({"type_id"}, fc);

        // for (int32_t i = 0; i < fc->nbFields; i++)
        // {
        //     PLUGIN_VALIDATE(fc->fields[i].name != nullptr);
        //     std::string fieldName(fc->fields[i].name);

        //     if (fieldName.compare("type_id") == 0)
        //     {
        //         typeId = *static_cast<int32_t const*>(fc->fields[i].data);
        //     }
        //     if (fieldName.compare("bias") == 0)
        //     {
        //         bias.values = fc->fields[i].data;
        //         bias.count = fc->fields[i].length;
        //         bias.type = fieldTypeToDataType(fc->fields[i].type);
        //     }
        // }

        // if (typeId < 0 || typeId > 3)
        // {
        //     gLogError << "GeluPluginCreator: invalid typeId " << typeId << std::endl;
        //     return nullptr;
        // }

        return plugin; //new GeluPlugin(name, static_cast<DataType>(typeId), bias);
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

IPluginV2* GeluPluginCreator::deserializePlugin(
    char const* name, void const* serialData, size_t serialLength) noexcept
{
    // This object will be deleted when the network is destroyed, which will
    // call GeluPlugin::destroy()
    try
    {
        return new GeluPlugin(name, serialData, serialLength);
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

void GeluPluginCreator::setPluginNamespace(char const* libNamespace) noexcept
{
    try
    {
        PLUGIN_VALIDATE(libNamespace != nullptr);
        mNamespace = libNamespace;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
}

char const* GeluPluginCreator::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

// #endif // CUDA_VERSION >= 10010

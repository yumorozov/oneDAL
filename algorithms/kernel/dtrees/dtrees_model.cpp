/* file: dtrees_model.cpp */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

/*
//++
//  Implementation of the class defining the decision trees model
//--
*/

#include "dtrees_model_impl.h"

using namespace daal::data_management;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace dtrees
{

namespace internal
{
Tree::~Tree()
{
}

ModelImpl::ModelImpl() : _nTree(0)
{
}

ModelImpl::~ModelImpl()
{
    destroy();
}

void ModelImpl::destroy()
{
    _serializationData.reset();
}

bool ModelImpl::reserve(const size_t nTrees)
{
    if(_serializationData.get())
        return false;
    _nTree.set(0);
    _serializationData.reset(new DataCollection());
    _serializationData->resize(nTrees);

    _impurityTables.reset(new DataCollection());
    _impurityTables->resize(nTrees);

    _nNodeSampleTables.reset(new DataCollection());
    _nNodeSampleTables->resize(nTrees);

    return _serializationData.get();
}

bool ModelImpl::resize(const size_t nTrees)
{
    if(_serializationData.get())
        return false;
    _nTree.set(0);
    _serializationData.reset(new DataCollection(nTrees));
    _impurityTables.reset(new DataCollection(nTrees));
    _nNodeSampleTables.reset(new DataCollection(nTrees));
    return _serializationData.get();
}

void ModelImpl::clear()
{
    if(_serializationData.get())
        _serializationData.reset();

    if(_impurityTables.get())
        _impurityTables.reset();

    if(_nNodeSampleTables.get())
        _nNodeSampleTables.reset();

    _nTree.set(0);
}

void MemoryManager::destroy()
{
    for(size_t i = 0; i < _aChunk.size(); ++i)
        daal_free(_aChunk[i]);
    _aChunk.clear();
    _posInChunk = 0;
    _iCurChunk = -1;
}

void* MemoryManager::alloc(size_t nBytes)
{
    DAAL_ASSERT(nBytes <= _chunkSize);
    size_t pos = 0; //pos in the chunk to allocate from
    if((_iCurChunk >= 0) && (_posInChunk + nBytes <= _chunkSize))
    {
        //allocate from the current chunk
        pos = _posInChunk;
    }
    else
    {
        if(!_aChunk.size() || _iCurChunk + 1 >= _aChunk.size())
        {
            //allocate a new chunk
            DAAL_ASSERT(_aChunk.size() ? _iCurChunk >= 0 : _iCurChunk < 0);
            byte* ptr = (byte*)services::daal_malloc(_chunkSize);
            if(!ptr)
                return nullptr;
            _aChunk.push_back(ptr);
        }
        //there are free chunks, make next available a current one and allocate from it
        _iCurChunk++;
        pos = 0;
        _posInChunk = 0;
    }
    //allocate from the current chunk
    _posInChunk += nBytes;
    return _aChunk[_iCurChunk] + pos;
}

void MemoryManager::reset()
{
    _iCurChunk = -1;
    _posInChunk = 0;
}

} // namespace internal
} // namespace dtrees
} // namespace algorithms
} // namespace daal

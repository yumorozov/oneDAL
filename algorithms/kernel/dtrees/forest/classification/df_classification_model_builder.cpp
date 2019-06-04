/* file: df_classification_model_builder.cpp */
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
//  Implementation of the class defining the decision forest classification model builder
//--
*/

#include "algorithms/decision_forest/decision_forest_classification_model_builder.h"
#include "algorithms/decision_forest/decision_forest_classification_model.h"
#include "../../dtrees_model_impl.h"
#include "df_classification_model_impl.h"

using namespace daal::data_management;
using namespace daal::services;
using namespace daal::algorithms::dtrees::internal;

namespace daal
{
namespace algorithms
{
namespace decision_forest
{
namespace classification
{
namespace interface1
{
#define __NODE_RESERVED_ID -2
#define __NODE_FREE_ID -3
#define __N_CHILDS 2

decision_forest::classification::internal::ModelImpl& getModelRef(ModelPtr& modelPtr)
{
    decision_forest::classification::internal::ModelImpl* modelImplPtr = dynamic_cast<decision_forest::classification::internal::ModelImpl*>(modelPtr.get());
    DAAL_ASSERT(modelImplPtr);
    return *modelImplPtr;
}

services::Status ModelBuilder::initialize(size_t nClasses, size_t nTrees)
{
    services::Status s;
    _model.reset(new decision_forest::classification::internal::ModelImpl());
    decision_forest::classification::internal::ModelImpl& modelImplRef = getModelRef(_model);

    modelImplRef.resize(nTrees);
    modelImplRef._impurityTables.reset();
    modelImplRef._nNodeSampleTables.reset();
    modelImplRef._nTree.set(nTrees);
    return s;
}

services::Status ModelBuilder::createTreeInternal(size_t nNodes, TreeId& resId)
{
    decision_forest::classification::internal::ModelImpl& modelImplRef = getModelRef(_model);
    Status s;
    if (nNodes == 0)
    {
        return Status(ErrorID::ErrorIncorrectParameter);
    }

    TreeId treeId = 0;
    const SerializationIface* isEmptyTreeTable = (*(modelImplRef._serializationData))[treeId].get();
    const size_t nTrees = (*(modelImplRef._serializationData)).size();
    while(isEmptyTreeTable && treeId < nTrees)
    {
        treeId++;
        isEmptyTreeTable = (*(modelImplRef._serializationData))[treeId].get();
    }
    if (treeId == nTrees)
        return Status(ErrorID::ErrorIncorrectParameter);

    services::SharedPtr<DecisionTreeTable> treeTablePtr(new DecisionTreeTable(nNodes));//DecisionTreeTable* const treeTable = new DecisionTreeTable(nNodes);
    const size_t nRows = treeTablePtr->getNumberOfRows();
    DecisionTreeNode* const pNodes = (DecisionTreeNode*)treeTablePtr->getArray();
    pNodes[0].featureIndex = __NODE_RESERVED_ID;
    pNodes[0].leftIndexOrClass = 0;
    pNodes[0].featureValueOrResponse = 0;

    for(size_t i = 1; i < nRows; i++)
    {
        pNodes[i].featureIndex = __NODE_FREE_ID;
        pNodes[i].leftIndexOrClass = 0;
        pNodes[i].featureValueOrResponse = 0;
    }

    (*(modelImplRef._serializationData))[treeId] = treeTablePtr;

    resId = treeId;
    return s;
}

services::Status ModelBuilder::addLeafNodeInternal(TreeId treeId, NodeId parentId, size_t position, size_t classLabel, NodeId& res)
{
    decision_forest::classification::internal::ModelImpl& modelImplRef = getModelRef(_model);

    Status s;
    if (treeId > (*(modelImplRef._serializationData)).size())
    {
        return Status(ErrorID::ErrorIncorrectParameter);
    }
    if (position != 0 && position != 1)
    {
        return Status(ErrorID::ErrorIncorrectParameter);
    }

    const DecisionTreeTable* const pTreeTable = static_cast<DecisionTreeTable*>((*(modelImplRef._serializationData))[treeId].get());
    DAAL_CHECK(pTreeTable, ErrorID::ErrorNullPtr);

    const size_t nRows = pTreeTable->getNumberOfRows();

    DecisionTreeNode* const aNode = (DecisionTreeNode*)pTreeTable->getArray();
    NodeId nodeId = 0;
    if (parentId == noParent)
    {
        aNode[0].featureIndex = -1;
        aNode[0].leftIndexOrClass = classLabel;
        aNode[0].featureValueOrResponse = 0;
        nodeId = 0;
    }
    else if (aNode[parentId].featureIndex < 0)
    {
        return Status(ErrorID::ErrorIncorrectParameter);
    }
    else
    {
        /*if not leaf, and parent has child already*/
        if ((aNode[parentId].leftIndexOrClass > 0) && (position == 1))
        {
            const NodeId reservedId = aNode[parentId].leftIndexOrClass + 1;
            nodeId = reservedId;
            if (aNode[reservedId].featureIndex == __NODE_RESERVED_ID)
            {
                aNode[nodeId].featureIndex = -1;
                aNode[nodeId].leftIndexOrClass = classLabel;
                aNode[nodeId].featureValueOrResponse = 0;
            }
        }
        else if ((aNode[parentId].leftIndexOrClass > 0) && (position == 0))
        {
            const NodeId reservedId = aNode[parentId].leftIndexOrClass;
            nodeId = reservedId;
            if (aNode[reservedId].featureIndex == __NODE_RESERVED_ID)
            {
                aNode[nodeId].featureIndex = -1;
                aNode[nodeId].leftIndexOrClass = classLabel;
                aNode[nodeId].featureValueOrResponse = 0;
            }
        }
        else if ((aNode[parentId].leftIndexOrClass == 0) && (position == 0))
        {
            size_t i;
            for(i = parentId + 1; i < nRows; i++)
            {
                if (aNode[i].featureIndex == __NODE_FREE_ID)
                {
                    nodeId = i;
                    break;
                }
            }
            /* no space left */
            if(i == nRows)
            {
                return Status(ErrorID::ErrorIncorrectParameter);
            }
            aNode[nodeId].featureIndex = -1;
            aNode[nodeId].leftIndexOrClass = classLabel;
            aNode[nodeId].featureValueOrResponse = 0;

            aNode[parentId].leftIndexOrClass = nodeId;

            if (((nodeId + 1) < nRows) && (aNode[nodeId+1].featureIndex == __NODE_FREE_ID))
            {
                    aNode[nodeId+1].featureIndex = __NODE_RESERVED_ID;
            }
            else
            {
                return Status(ErrorID::ErrorIncorrectParameter);
            }
        }
        else if ((aNode[parentId].leftIndexOrClass == 0) && (position == 1))
        {
            NodeId leftEmptyId = 0;
            size_t i;
            for(i = parentId + 1; i < nRows; i++)
            {
                if (aNode[i].featureIndex == __NODE_FREE_ID)
                {
                    leftEmptyId = i;
                    break;
                }
            }
            /*if no free nodes leftBound is not initialized and no space left*/
            if (i == nRows)
            {
                return Status(ErrorID::ErrorIncorrectParameter);
            }

            aNode[leftEmptyId].featureIndex = __NODE_RESERVED_ID;

            aNode[parentId].leftIndexOrClass = leftEmptyId;
            nodeId = leftEmptyId + 1;
            if (nodeId < nRows)
            {
                aNode[nodeId].featureIndex = -1;
                aNode[nodeId].leftIndexOrClass = classLabel;
                aNode[nodeId].featureValueOrResponse = 0;
            }
            else
            {
                return Status(ErrorID::ErrorIncorrectParameter);
            }
        }

    }
    res = nodeId;
    return s;
}

services::Status ModelBuilder::addSplitNodeInternal(TreeId treeId, NodeId parentId, size_t position, size_t featureIndex, double featureValue, NodeId& res)
{
    Status s;
    decision_forest::classification::internal::ModelImpl& modelImplRef = getModelRef(_model);
    if (treeId > (*(modelImplRef._serializationData)).size())
    {
        return Status(ErrorID::ErrorIncorrectParameter);
    }
    if (position != 0 && position != 1)
    {
        return Status(ErrorID::ErrorIncorrectParameter);
    }

    const DecisionTreeTable* const pTreeTable = static_cast<DecisionTreeTable*>((*(modelImplRef._serializationData))[treeId].get());
    DAAL_CHECK(pTreeTable, ErrorID::ErrorNullPtr);
    const size_t nRows = pTreeTable->getNumberOfRows();

    DecisionTreeNode* const aNode = (DecisionTreeNode*)pTreeTable->getArray();
    NodeId nodeId = 0;
    if (parentId == noParent)
    {
        aNode[0].featureIndex = featureIndex;
        aNode[0].leftIndexOrClass = 0;
        aNode[0].featureValueOrResponse = featureValue;
        nodeId = 0;
    }
    else if (aNode[parentId].featureIndex < 0)
    {

        return Status(ErrorID::ErrorIncorrectParameter);
    }
    else
    {
        /*if not leaf, and parent has child already*/
        if ((aNode[parentId].leftIndexOrClass > 0) && (position == 1))
        {
            const NodeId reservedId = aNode[parentId].leftIndexOrClass + 1;
            nodeId = reservedId;
            if (aNode[reservedId].featureIndex == __NODE_RESERVED_ID)
            {
                aNode[nodeId].featureIndex = featureIndex;
                aNode[nodeId].leftIndexOrClass = 0;
                aNode[nodeId].featureValueOrResponse = featureValue;
            }
        }

        if ((aNode[parentId].leftIndexOrClass > 0) && (position == 0))
        {
            const NodeId reservedId = aNode[parentId].leftIndexOrClass;
            nodeId = reservedId;
            if (aNode[reservedId].featureIndex == __NODE_RESERVED_ID)
            {
                aNode[nodeId].featureIndex = featureIndex;
                aNode[nodeId].leftIndexOrClass = 0;
                aNode[nodeId].featureValueOrResponse = featureValue;
            }
        }

        if ((aNode[parentId].leftIndexOrClass == 0) && (position == 0))
        {
            size_t i;
            for(i = parentId + 1; i < nRows; i++)
            {
                if (aNode[i].featureIndex == __NODE_FREE_ID)
                {
                    nodeId = i;
                    break;
                }
            }
            /* no space left */
            if(i == nRows)
            {
                return Status(ErrorID::ErrorIncorrectParameter);
            }
            aNode[nodeId].featureIndex = featureIndex;
            aNode[nodeId].leftIndexOrClass = 0;
            aNode[nodeId].featureValueOrResponse = featureValue;

            aNode[parentId].leftIndexOrClass = nodeId;

            if (((nodeId + 1) < nRows) && (aNode[nodeId+1].featureIndex == __NODE_FREE_ID))
            {
                    aNode[nodeId+1].featureIndex = __NODE_RESERVED_ID;
            }
            else
            {
                return Status(ErrorID::ErrorIncorrectParameter);
            }
        }

        if ((aNode[parentId].leftIndexOrClass == 0) && (position == 1))
        {
            NodeId leftEmptyId = 0;
            size_t i;
            for(i = parentId + 1; i < nRows; i++)
            {
                if (aNode[i].featureIndex == __NODE_FREE_ID)
                {
                    leftEmptyId = i;
                    break;
                }
            }
            /*if no free nodes leftBound is not initialized and no space left*/
            if (i == nRows)
            {
                return Status(ErrorID::ErrorIncorrectParameter);
            }

            aNode[leftEmptyId].featureIndex = __NODE_RESERVED_ID;

            aNode[parentId].leftIndexOrClass = leftEmptyId;
            nodeId = leftEmptyId + 1;
            if (nodeId < nRows)
            {
                aNode[nodeId].featureIndex = featureIndex;
                aNode[nodeId].leftIndexOrClass = 0;
                aNode[nodeId].featureValueOrResponse = featureValue;
            }
            else
            {
                return Status(ErrorID::ErrorIncorrectParameter);
            }
        }

    }
    res = nodeId;
    return s;
}

} // namespace interface1
} // namespace classification
} // namespace decision_forest
} // namespace algorithms
} // namespace daal

//
// Copyright (c) Stanislav Grigoriev. All rights reserved.
// grigorievstas9@gmail.com 
// https://github.com/StanislavGrigoriev/EasyCNTK
//
// Copyright (c) Microsoft. All rights reserved.
//
// Licensed under the MIT license. See LICENSE.txt file in the project root for full license information.
//
using CNTK;
using System;
using System.Collections.Generic;
using System.Linq;

namespace EasyCNTK.LossFunctions
{
    public sealed class WeightedBinaryCrossEntropyError<T> : Loss
    {
        private IList<double> _weights;
        
        public WeightedBinaryCrossEntropyError(IList<double> weights)
        {
            _weights = weights ?? throw new ArgumentNullException("weights");            
        }
        public override Function GetLoss(Variable prediction, Variable targets, DeviceDescriptor device)
        {
            Variable weights = null;
            var uid = Guid.NewGuid().ToString();
            if (targets.DataType == DataType.Double)
            {
                weights = new Variable(targets.Shape, VariableKind.Constant, targets.DataType, new NDArrayView(targets.Shape, _weights.ToArray(), device), false, null, false, "weights", uid);
            }
            if (targets.DataType == DataType.Float)
            {
                weights = new Variable(targets.Shape, VariableKind.Constant, targets.DataType, new NDArrayView(targets.Shape, _weights.ToArray(), device), false, null, false, "weights", uid);
            }            

            return CNTKLib.WeightedBinaryCrossEntropy(prediction, targets, weights);
        }
    }
}

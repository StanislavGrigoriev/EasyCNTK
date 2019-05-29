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

namespace EasyCNTK.LossFunctions
{
    public sealed class CrossEntropyWithSoftmaxError : Loss
    {
        private Axis _axis;
        public CrossEntropyWithSoftmaxError()
        {
            _axis = null;
        }
        public CrossEntropyWithSoftmaxError(int axisNumber)
        {
            _axis = new Axis(axisNumber);            
        }
        public override Function GetLoss(Variable prediction, Variable targets, DeviceDescriptor device)
        {
            if (_axis != null)
            {
                return CNTKLib.CrossEntropyWithSoftmax(prediction, targets, _axis);
            }
            return CNTKLib.CrossEntropyWithSoftmax(prediction, targets);
        }
    }
}

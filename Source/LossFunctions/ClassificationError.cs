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
    public sealed class ClassificationError : Loss
    {
        private int _numberAxis;

        public ClassificationError()
        {
            _numberAxis = -1;
        }
        public ClassificationError(int numberAxis)
        {
            _numberAxis = numberAxis;
        }
        public override Function GetLoss(Variable prediction, Variable targets, DeviceDescriptor device)
        {
            if (_numberAxis == -1)
            {
                return CNTKLib.ClassificationError(prediction, targets);
            }
            return CNTKLib.ClassificationError(prediction, targets, new Axis(_numberAxis));
        }
    }
}

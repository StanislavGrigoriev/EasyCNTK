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

namespace EasyCNTK.LossFunctions
{
    public sealed class BinaryClassificationError : Loss
    {
        private double _thresholdValue;
        public BinaryClassificationError(double threshold = 0.5)
        {
            if (threshold <= 0 || threshold >= 1) throw new ArgumentOutOfRangeException("threshold", "Порог должен быть в диапазоне: (0;1)");
            _thresholdValue = threshold;
        }
        public override Function GetLoss(Variable prediction, Variable targets, DeviceDescriptor device)
        {
            var threshold = new Constant(prediction.Shape, prediction.DataType, _thresholdValue, device);

            var predictionLabel = CNTKLib.Less(prediction, threshold);
            var loss = CNTKLib.Equal(predictionLabel, targets);

            return loss;
        }
    }
}

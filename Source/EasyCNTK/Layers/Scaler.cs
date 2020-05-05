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

namespace EasyCNTK.Layers
{
    /// <summary>
    /// Реализует масштабирование выхода путем перемножения на константу
    /// </summary>
    public sealed class Scaler : Layer
    {
        double _factor;
        /// <summary>
        /// Масштабирует выход перемножая каждый элемент на константу
        /// </summary>
        /// <param name="factor">Констатнта, на которую перемножается выход</param>
        public Scaler(double factor)
        {
            _factor = factor;
        }
        public override Function Create(Function input, DeviceDescriptor device)
        {
            var factor = Constant.Scalar(input.Output.DataType, _factor, device);
            var scaled = CNTKLib.ElementTimes(factor, input);
            return scaled;
        }

        public override string GetDescription()
        {
            return $"Scale({_factor})";
        }
    }
}

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
    /// Реализует слой самостаблизации для выбора оптимальной скорости обучения. Источник: https://github.com/Microsoft/CNTK/blob/release/latest/Examples/TrainingCSharp/Common/LSTMSequenceClassifier.cs
    /// </summary>
    public sealed class SelfStabilization<T> : Layer
    {
        private string _name;
        private static Function selfStabilize(Function input, DeviceDescriptor device, string name)
        {
            DataType dataType = typeof(T) == typeof(float) ? DataType.Float : DataType.Double;
           
            Constant f = Constant.Scalar(dataType, 4.0, device);
            Constant fInv = Constant.Scalar(dataType, 1.0 / 4.0);

            var beta = CNTKLib.ElementTimes(
                fInv,
                CNTKLib.Log(
                    Constant.Scalar(f.DataType, 1.0) +
                    CNTKLib.Exp(CNTKLib.ElementTimes(f,
                        new Parameter(new NDShape(), f.DataType, 0.99537863 /* 1/f*ln(e^f-1)*/, device, "alpha")))),
                "beta");
            return Function.Alias(CNTKLib.ElementTimes(beta, input), name);
        }
        /// <summary>
        /// Создает слой самостабилизации для выбора оптимальной скорости обучения.
        /// </summary>
        /// <param name="input"></param>
        /// <param name="device"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        public static Function Build(Function input, DeviceDescriptor device, string name)
        {
            return selfStabilize(input, device, name);
        }
        public override Function Create(Function input, DeviceDescriptor device)
        {
            return selfStabilize(input, device, _name);
        }
        /// <summary>
        /// Создает слой самостабилизации для выбора оптимальной скорости обучения.
        /// </summary>
        /// <param name="name"></param>
        public SelfStabilization(string name = "SelfStabilizer")
        {
            _name = name;
        }
        public override string GetDescription()
        {
            return "SS";
        }
    }
}

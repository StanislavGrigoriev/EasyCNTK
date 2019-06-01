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
    /// <summary>
    /// Функция ошибки при бинарной, одноклассовой и многоклассовой классификации. Допускается использование с Softmax выходом, при этом создается условие классификации: класс должен иметь вероятность выше заданного порога, иначе он не будет классифицирован.
    /// </summary>
    public sealed class ClassificationError : Loss
    {
        private double _thresholdValue;
        /// <summary>
        /// Функция ошибки при бинарной, одноклассовой и многоклассовой классификации. Допускается использование с Softmax выходом, при этом создается условие классификации: класс должен иметь вероятность выше заданного порога, иначе он не будет классифицирован.
        /// </summary>
        /// <param name="threshold">Пороговое значение для действительного значения выхода нейросети, ниже которого класс не распознается. Другими словами - это минимальная вероятность, которую должен выдать классификатор для конкретного класса, чтобы этот класс был учтен как распознанный.</param>
        public ClassificationError(double threshold = 0.5)
        {
            if (threshold <= 0 || threshold >= 1) throw new ArgumentOutOfRangeException("threshold", "Порог должен быть в диапазоне: (0;1)");
            _thresholdValue = threshold;
        }
        public override Function GetLoss(Variable prediction, Variable targets, DeviceDescriptor device)
        {
            var threshold = new Constant(prediction.Shape, prediction.DataType, _thresholdValue, device);

            var predictionLabel = CNTKLib.Less(threshold, prediction);
            var loss = CNTKLib.NotEqual(predictionLabel, targets);

            return loss;
        }
    }
}

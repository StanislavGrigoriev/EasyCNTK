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
    /// <summary>
    /// Функция ошибки для одноклассовой классификации при использовании Softmax выхода 
    /// </summary>
    public sealed class SoftmaxClassificationError : Loss
    {
        private int _numberAxis;

        /// <summary>
        /// Функция ошибки для одноклассовой классификации при использовании Softmax выхода 
        /// </summary>
        public SoftmaxClassificationError()
        {
            _numberAxis = -1;
        }
        /// <summary>
        /// Функция ошибки для одноклассовой классификации при использовании Softmax выхода 
        /// </summary>
        /// <param name="numberAxis">Номер оси, вдоль которой применяется оценка классификации</param>
        public SoftmaxClassificationError(int numberAxis)
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

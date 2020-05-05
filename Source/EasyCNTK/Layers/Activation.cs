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
using EasyCNTK.ActivationFunctions;

namespace EasyCNTK.Layers
{
    /// <summary>
    /// Реализует применение функции активации к предыдущему слою
    /// </summary>
    public sealed class Activation : Layer
    {
        private ActivationFunction _activation;

        /// <summary>
        /// Создает слой, применяющий функцию активации к предыдущему слою
        /// </summary>
        /// <param name="activationFunction"></param>
        public Activation(ActivationFunction activationFunction)
        {
            _activation = activationFunction;
        }
        public override Function Create(Function input, DeviceDescriptor device)
        {
            return _activation?.ApplyActivationFunction(input, device) ?? input;
        }

        public override string GetDescription()
        {
            return $"Activation[{_activation?.GetDescription() ?? "None"}]";
        }
    }
}

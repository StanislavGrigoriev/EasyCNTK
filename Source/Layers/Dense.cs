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
    /// Реализует полносвязный слой с заданной функцией активации
    /// </summary>
    public sealed class Dense: Layer
    {
        private int _outputDim;
        private ActivationFunction _activationFunction;
        private string _name;

        /// <summary>
        /// Создает полносвязный слой с заданной функцией активации
        /// </summary>
        /// <param name="input">Входная переменная(слой) заданной разрядности</param>
        /// <param name="outputDim">Выходная разрядность(кол-во нейронов)</param>
        /// <param name="activationFunction">Функция активации</param>
        /// <param name="device">Устройство на котором производится расчет</param>
        /// <param name="name">Имя слоя</param>
        /// <returns></returns>
        private static Function createFullyConnectedLinearLayer(Variable input, int outputDim, ActivationFunction activationFunction, DeviceDescriptor device, string name)
        {
            var dataType = input.DataType;            
            var inputDim = input.Shape[0];
            var weight   = new Parameter(new int[] { outputDim, inputDim }, dataType, CNTKLib.GlorotUniformInitializer(
                CNTKLib.DefaultParamInitScale,
                    CNTKLib.SentinelValueForInferParamInitRank,
                    CNTKLib.SentinelValueForInferParamInitRank, 1), device);
            var bias                    = new Parameter(new int[] { outputDim }, dataType, 0, device);
            var fullyConnected          = CNTKLib.Times(weight, input) + bias;
            var activatedFullyConnected = activationFunction?.ApplyActivationFunction(fullyConnected, device) ?? fullyConnected;
            return Function.Alias(activatedFullyConnected, name);
        }
        /// <summary>
        /// Создает полносвязный слой с заданной функцией активации
        /// </summary>
        /// <param name="input">Входная переменная(слой) заданной разрядности</param>
        /// <param name="outputDim">Выходная разрядность(кол-во нейронов)</param>
        /// <param name="activationFunction">Функция активации</param>
        /// <param name="device">Устройство на котором производится расчет</param>
        /// <param name="name">Имя слоя</param>
        /// <returns></returns>
        public static Function Build(Function input, int outputDim, ActivationFunction activationFunction, DeviceDescriptor device, string name = "Dense")
        {
            return createFullyConnectedLinearLayer(input, outputDim, activationFunction, device, name);
        }        
        public override Function Create(Function input, DeviceDescriptor device)
        {
            return createFullyConnectedLinearLayer(input, _outputDim, _activationFunction, device, _name);
        }
        /// <summary>
        /// Создает полносвязный слой с заданной функцией активации
        /// </summary>
        /// <param name="outputDimension">Выходная разрядность(кол-во нейронов)</param>
        /// <param name="activationFunction">Функция активации, null если не требуется</param>
        /// <param name="name">Имя слоя</param>
        public Dense(int outputDimension, ActivationFunction activationFunction, string name = "Dense")
        {
            _outputDim = outputDimension;
            _activationFunction = activationFunction;
            _name = name;
        }  

        public override string GetDescription()
        {
            return $"{_outputDim}[{_activationFunction?.GetDescription()}]";
        }
    }
}

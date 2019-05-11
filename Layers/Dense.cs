using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
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
            if (input.Shape.Rank != 1)
            {
                // если данные не одномерные разворачиваем входной тензор в вектор
                int newDim = input.Shape.Dimensions.Aggregate((d1, d2) => d1 * d2);
                input = CNTKLib.Reshape(input, new int[] { newDim });
            }

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

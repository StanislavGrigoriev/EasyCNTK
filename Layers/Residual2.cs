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
    /// Реализует остаточный слой с 2 внутренними слоями
    /// </summary>
    public sealed class Residual2 : Layer
    {
        private int _outputDim;
        private ActivationFunction _activationFunction;
        private string _name;

        private static Function createResidualLayer2(Function input, int outputDimension, ActivationFunction activationFunction, DeviceDescriptor device, string name)
        {
            var dataType = input.Output.DataType;

            if (input.Output.Shape.Rank != 1)
            {
                int newDim = input.Output.Shape.Dimensions.Aggregate((d1, d2) => d1 * d2);
                input = CNTKLib.Reshape(input, new int[] { newDim });
            }
            //проброс входа мимо 1 слоя    
            var forwarding = input;
            if (outputDimension != input.Output.Shape[0])
            {
                var scales = new Parameter(new int[] { outputDimension, input.Output.Shape[0] }, dataType, CNTKLib.UniformInitializer(CNTKLib.DefaultParamInitScale), device);
                forwarding = CNTKLib.Times(scales, forwarding);
            }

            //создание 1 слоя
            input = Dense.Build(input, outputDimension, activationFunction, device, "");
            //создание 2 слоя без функции активации
            input = Dense.Build(input, outputDimension, null, device, "");
            //соединение с проброшенным входом
            input = CNTKLib.Plus(input, forwarding);

            input = activationFunction?.ApplyActivationFunction(input, device) ?? input;
            return Function.Alias(input, name);
        }
        /// <summary>
        /// Создает остаточный слой с 2 внутренними слоями
        /// </summary>
        /// <typeparam name="TElement">Тип данных. Поддерживается <seealso cref="float"/>, <seealso cref="double"/></typeparam>
        /// <param name="input">Вход</param>
        /// <param name="outputDimension">Разрядность выходного слоя</param>
        /// <param name="activationFunction">Функция активации, null если не требуется</param>
        /// <param name="device">Устройство для расчетов</param>
        /// <param name="name"></param>
        /// <returns></returns>
        public static Function Build(Function input, int outputDimension, ActivationFunction activationFunction, DeviceDescriptor device, string name = "Residual2")
        {
            return createResidualLayer2(input, outputDimension, activationFunction, device, name);
        }
        /// <summary>
        /// Создает остаточный слой с 2 внутренними слоями
        /// </summary>
        /// <param name="outputDimension">Разрядность выходного слоя</param>
        /// <param name="activationFunction">Функция активации, null если не требуется</param>
        /// <param name="name"></param>
        public Residual2(int outputDimension, ActivationFunction activationFunction, string name = "Residual2")
        {
            _outputDim = outputDimension;
            _activationFunction = activationFunction;
            _name = name;
        }
        public override Function Create(Function input, DeviceDescriptor device)
        {
            return createResidualLayer2(input, _outputDim, _activationFunction, device, _name);
        }
     
        public override string GetDescription()
        {
            return $"Res2({_outputDim})[{_activationFunction?.GetDescription()}]";
        }
    }
}

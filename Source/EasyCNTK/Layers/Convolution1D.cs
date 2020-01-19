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
    /// Реализует сверточный слой в одномерном пространстве
    /// </summary>
    public class Convolution1D : Layer
    {
        private int _kernelWidth;        
        private int _inputChannelsCount;
        private int _outChannelsCount;
        private int _hStride;       
        private Padding _padding;
        private ActivationFunction _activationFunction;
        private string _name;
        /// <summary>
        /// Добавляет одномерный сверточный слой с разным числом каналов. Если предыдущий слой имеет не одномерный/двумерный выход, выбрасывается исключение
        /// </summary>
        /// <param name="input"></param>
        /// <param name="kernelWidth">Ширина ядра свертки</param>        
        /// <param name="device">Устройство для расчетов</param>
        /// <param name="inputChannelsCount">Количество каналов входного изображения, для черно-белого изображения равно 1, для цветного (RGB) равно 3</param>
        /// <param name="outChannelsCount">Количество каналов выходной ячейки (разрядность выходной ячейки после свертки). Количество каналов <paramref name="inputChannelsCount"/> следующего слоя должно равнятся числу выходных каналов текущего слоя. </param>
        /// <param name="activationFunction">Функция активации для выходного слоя. Если не требуется - передать null</param>
        /// <param name="hStride">Шаг спещения окна свертки по горизонтали</param>        
        /// <param name="padding">Заполнение при использовании сверток</param>
        /// <param name="name"></param>
        public static Function Build(Variable input, int kernelWidth, DeviceDescriptor device, int inputChannelsCount = 1, int outChannelsCount = 1, int hStride = 1, Padding padding = Padding.Valid, ActivationFunction activationFunction = null, string name = "Conv1D")
        {
            bool[] paddingVector = null;
            if (padding == Padding.Valid)
            {
                paddingVector = new bool[] { false, false };
            }
            if (padding == Padding.Same)
            {
                paddingVector = new bool[] { true, false };
            }

            var convMap = new Parameter(new int[] { kernelWidth, inputChannelsCount, outChannelsCount }, input.DataType, CNTKLib.GlorotUniformInitializer(), device);
            var convolution = CNTKLib.Convolution(convMap, input, new int[] { hStride, inputChannelsCount }, new bool[] { true }, paddingVector);
            var activatedConvolution = activationFunction?.ApplyActivationFunction(convolution, device) ?? convolution;

            return Function.Alias(activatedConvolution, name);
        }

        public override Function Create(Function input, DeviceDescriptor device)
        {
            return Build(input, _kernelWidth, device, _inputChannelsCount, _outChannelsCount, _hStride, _padding, _activationFunction, _name);
        }
        /// <summary>
        /// Добавляет одномерный сверточный слой с разным числом каналов. Если предыдущий слой имеет не одномерный/двумерный выход, выбрасывается исключение
        /// </summary>
        /// <param name="kernelWidth">Ширина ядра свертки</param>        
        /// <param name="inputChannelsCount">Количество каналов входного изображения, для черно-белого изображения равно 1, для цветного (RGB) равно 3</param>
        /// <param name="outChannelsCount">Разрядность выходной ячейки после свертки</param>
        /// <param name="activationFunction">Функция активации для выходного слоя. Если не требуется - передать null</param>
        /// <param name="hStride">Шаг спещения окна свертки по горизонтали</param>        
        /// <param name="padding">Заполнение при использовании сверток</param>
        /// <param name="name"></param>
        public Convolution1D(int kernelWidth, int inputChannelsCount = 1, int outChannelsCount = 1, int hStride = 1, Padding padding = Padding.Valid, ActivationFunction activationFunction = null, string name = "Conv2D")
        {
            _kernelWidth = kernelWidth;
            _inputChannelsCount = inputChannelsCount;
            _outChannelsCount = outChannelsCount;
            _hStride = hStride;
            _padding = padding;
            _activationFunction = activationFunction;
            _name = name;
        }
        public override string GetDescription()
        {
            return $"Conv1D(K={_kernelWidth}x{_inputChannelsCount}S={_hStride}P={_padding})[{_activationFunction?.GetDescription()}]";
        }
    }
}

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
    /// Реализует сверточный слой в трехмерном просранстве
    /// </summary>
    public sealed class Convolution3D: Layer
    {
        private int _kernelWidth;
        private int _kernelHeight;
        private int _kernelDepth;
        private int _inputChannelsCount;
        private int _outChannelsCount;
        private int _hStride;
        private int _vStride;
        private int _dStride;
        private Padding _padding;
        private ActivationFunction _activationFunction;
        private string _name;
        /// <summary>
        /// Добавляет трехмерный сверточный слой с разным числом каналов. Если предыдущий слой имеет не трехмерный выход, выбрасывается исключение
        /// </summary>
        /// <param name="input"></param>
        /// <param name="kernelWidth">Ширина ядра свертки (x ось куба)</param>
        /// <param name="kernelHeight">Высота ядра свертки (y ось куба)</param>
        /// <param name="kernelDepth">Глубина ядра свертки (z ось куба)</param>
        /// <param name="device">Устройство для расчетов</param>
        /// <param name="inputChannelsCount">Количество каналов входного 3D изображения(куба), для черно-белого изображения равно 1, для цветного (RGB) равно 3</param>
        /// <param name="outChannelsCount">Количество каналов выходной ячейки (разрядность выходной ячейки после свертки). Количество каналов <paramref name="inputChannelsCount"/> следующего слоя должно равнятся числу выходных каналов текущего слоя. </param>
        /// <param name="activationFunction">Функция активации для выходного слоя. Если не требуется - передать null</param>
        /// <param name="hStride">Шаг спещения окна свертки по оси X</param>
        /// <param name="vStride">Шаг смещения окна свертки по оси Y</param>
        /// <param name="dStride">Шаг смещения окна свертки по оси Z</param>
        /// <param name="padding">Заполнение при использовании сверток</param>
        /// <param name="name"></param>
        public static Function Build(Variable input, int kernelWidth, int kernelHeight, int kernelDepth, DeviceDescriptor device, int inputChannelsCount = 1, int outChannelsCount = 1, int hStride = 1, int vStride = 1, int dStride = 1, Padding padding = Padding.Valid, ActivationFunction activationFunction = null, string name = "Conv3D")
        {
            bool[] paddingVector = null;
            if (padding == Padding.Valid)
            {
                paddingVector = new bool[] { false, false, false, false };
            }
            if (padding == Padding.Same)
            {
                paddingVector = new bool[] { true, true, true, false };
            }

            var convMap = new Parameter(new int[] { kernelWidth, kernelHeight, kernelDepth, inputChannelsCount, outChannelsCount }, input.DataType, CNTKLib.GlorotUniformInitializer(), device);
            var convolution = CNTKLib.Convolution(convMap, input, new int[] { hStride, vStride, dStride, inputChannelsCount }, new bool[] { true }, paddingVector);
            var activatedConvolution = activationFunction?.ApplyActivationFunction(convolution, device) ?? convolution;

            return Function.Alias(activatedConvolution, name);
        }

        public override Function Create(Function input, DeviceDescriptor device)
        {
            return Build(input, _kernelWidth, _kernelHeight, _kernelDepth, device, _inputChannelsCount, _outChannelsCount, _hStride, _vStride, _dStride, _padding, _activationFunction, _name);
        }
        /// <summary>
        /// Добавляет трехмерный сверточный слой с разным числом каналов. Если предыдущий слой имеет не трехмерный выход, выбрасывается исключение
        /// </summary>
        /// <param name="kernelWidth">Ширина ядра свертки (x ось куба)</param>
        /// <param name="kernelHeight">Высота ядра свертки (y ось куба)</param>
        /// <param name="kernelDepth">Глубина ядра свертки (z ось куба)</param>      
        /// <param name="inputChannelsCount">Количество каналов входного 3D изображения(куба), для черно-белого изображения равно 1, для цветного (RGB) равно 3</param>
        /// <param name="outChannelsCount">Количество каналов выходной ячейки (разрядность выходной ячейки после свертки). Количество каналов <paramref name="inputChannelsCount"/> следующего слоя должно равнятся числу выходных каналов текущего слоя. </param>
        /// <param name="activationFunction">Функция активации для выходного слоя. Если не требуется - передать null</param>
        /// <param name="hStride">Шаг спещения окна свертки по оси X</param>
        /// <param name="vStride">Шаг смещения окна свертки по оси Y</param>
        /// <param name="dStride">Шаг смещения окна свертки по оси Z</param>
        /// <param name="padding">Заполнение при использовании сверток</param>
        /// <param name="name"></param>
        public Convolution3D(int kernelWidth, int kernelHeight, int kernelDepth, int inputChannelsCount = 1, int outChannelsCount = 1, int hStride = 1, int vStride = 1, int dStride = 1, Padding padding = Padding.Valid, ActivationFunction activationFunction = null, string name = "Conv3D")
        {
            _kernelWidth = kernelWidth;
            _kernelHeight = kernelHeight;
            _kernelDepth = kernelDepth;
            _inputChannelsCount = inputChannelsCount;
            _outChannelsCount = outChannelsCount;
            _hStride = hStride;
            _vStride = vStride;
            _dStride = dStride;
            _padding = padding;
            _activationFunction = activationFunction;
            _name = name;
        }
        public override string GetDescription()
        {
            return $"Conv3D(K={_kernelWidth}x{_kernelHeight}x{_kernelDepth}x{_inputChannelsCount}S={_hStride}x{_vStride}x{_dStride}P={_padding})[{_activationFunction?.GetDescription()}]";
        }
    }
}

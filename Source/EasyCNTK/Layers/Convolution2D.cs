﻿//
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
    /// Реализует сверточный слой в двумерном пространстве
    /// </summary>
    public sealed class Convolution2D : Layer
    {
        private int _kernelWidth;
        private int _kernelHeight;
        private int _inputChannelsCount;
        private int _outChannelsCount;
        private int _hStride;
        private int _vStride;
        private Padding _padding;
        private ActivationFunction _activationFunction;
        private string _name;
        /// <summary>
        /// Добавляет двумерный сверточный слой с разным числом каналов. Если предыдущий слой имеет не двумерный выход, выбрасывается исключение
        /// </summary>
        /// <param name="input"></param>
        /// <param name="kernelWidth">Ширина ядра свертки (столбцы в двумерной матрице/горизонталь изображения)</param>
        /// <param name="kernelHeight">Высота ядра свертки (строки в двумерной матрице/вертикаль изображения)</param>
        /// <param name="device">Устройство для расчетов</param>
        /// <param name="inputChannelsCount">Количество каналов входного изображения, для черно-белого изображения равно 1, для цветного (RGB) равно 3</param>
        /// <param name="outChannelsCount">Количество каналов выходной ячейки (разрядность выходной ячейки после свертки). Количество каналов <paramref name="inputChannelsCount"/> следующего слоя должно равнятся числу выходных каналов текущего слоя. </param>
        /// <param name="activationFunction">Функция активации для выходного слоя. Если не требуется - передать null</param>
        /// <param name="hStride">Шаг спещения окна свертки по горизонтали (по столбцам матрицы)</param>
        /// <param name="vStride">Шаг смещения окна свертки по вертикали (по строкам матрицы)</param>
        /// <param name="padding">Заполнение при использовании сверток</param>
        /// <param name="name"></param>
        public static Function Build(Variable input, int kernelWidth, int kernelHeight, DeviceDescriptor device, int inputChannelsCount = 1, int outChannelsCount = 1, int hStride = 1, int vStride = 1, Padding padding = Padding.Valid, ActivationFunction activationFunction = null, string name = "Conv2D")
        {
            bool[] paddingVector = null;
            if (padding == Padding.Valid)
            {
                paddingVector = new bool[] { false, false, false };
            }
            if (padding == Padding.Same)
            {
                paddingVector = new bool[] { true, true, false };
            }
            
            var convMap = new Parameter(new int[] { kernelWidth, kernelHeight, inputChannelsCount, outChannelsCount }, input.DataType, CNTKLib.GlorotUniformInitializer(), device);
            var convolution = CNTKLib.Convolution(convMap, input, new int[] { hStride, vStride, inputChannelsCount }, new bool[] { true }, paddingVector);
            var activatedConvolution = activationFunction?.ApplyActivationFunction(convolution, device) ?? convolution;

            return Function.Alias(activatedConvolution, name);
        }

        public override Function Create(Function input, DeviceDescriptor device)
        {
            return Build(input, _kernelWidth, _kernelHeight, device, _inputChannelsCount, _outChannelsCount, _hStride, _vStride, _padding, _activationFunction, _name);
        }
        /// <summary>
        /// Добавляет двумерный сверточный слой с разным числом каналов. Если предыдущий слой имеет не двумерный выход, выбрасывается исключение
        /// </summary>
        /// <param name="kernelWidth">Ширина ядра свертки (столбцы в двумерной матрице)</param>
        /// <param name="kernelHeight">Высота ядра свертки (строки в двумерной матрице)</param>
        /// <param name="inputChannelsCount">Количество каналов входного изображения, для черно-белого изображения равно 1, для цветного (RGB) равно 3</param>
        /// <param name="outChannelsCount">Разрядность выходной ячейки после свертки</param>
        /// <param name="activationFunction">Функция активации для выходного слоя. Если не требуется - передать null</param>
        /// <param name="hStride">Шаг спещения окна свертки по горизонтали (по столбцам матрицы)</param>
        /// <param name="vStride">Шаг смещения окна свертки по вертикали (по строкам матрицы)</param>
        /// <param name="padding">Заполнение при использовании сверток</param>
        /// <param name="name"></param>
        public Convolution2D(int kernelWidth, int kernelHeight, int inputChannelsCount = 1, int outChannelsCount = 1, int hStride = 1, int vStride = 1, Padding padding = Padding.Valid, ActivationFunction activationFunction = null, string name = "Conv2D")
        {
            _kernelWidth = kernelWidth;
            _kernelHeight = kernelHeight;
            _inputChannelsCount = inputChannelsCount;
            _outChannelsCount = outChannelsCount;
            _hStride = hStride;
            _vStride = vStride;
            _padding = padding;
            _activationFunction = activationFunction;
            _name = name;
        }
        public override string GetDescription()
        {
            return $"Conv2D(K={_kernelWidth}x{_kernelHeight}x{_inputChannelsCount}S={_hStride}x{_vStride}P={_padding})[{_activationFunction?.GetDescription()}]";
        }
    }
    /// <summary>
    /// Задает заполнение при использовании сверток
    /// </summary>
    public enum Padding
    {
        /// <summary>
        /// Заполнения краев нет (перемещения ядра свертки строго ограничено размерами изображения), изображение сворачивается по классике: n-f+1 x n-f+1
        /// </summary>
        Valid,
        /// <summary>
        /// Заполнения краев есть (перемещения ядра свертки выходит за границы изображения, лишняя часть дополняется нолями, выходное изображение остается того же размера что и до свертки), изображение сворачивается: n+2p-f+1 x n+2p-f+1
        /// </summary>
        Same
    }
}

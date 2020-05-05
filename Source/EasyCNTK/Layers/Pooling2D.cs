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
    /// Реализует пуллинг слой для двумерного вектора
    /// </summary>
    public sealed class Pooling2D : Layer
    {
        private int _poolingWindowWidth;
        private int _poolingWindowHeight;
        private int _hStride;
        private int _vStride;
        private PoolingType _poolingType;
        private Padding _padding;
        private string _name;
        /// <summary>
        /// Добавляет пуллинг слой для двумерного вектора. Если предыдущий слой имеет не двумерный выход, выбрасывается исключение
        /// </summary>
        /// <param name="poolingWindowWidth">Ширина окна пуллинга</param>
        /// <param name="poolingWindowHeight">Высота окна пуллинга</param>
        /// <param name="hStride">Шаг смещения окна пуллинга по горизонтали (по столбцам матрицы)</param>
        /// <param name="vStride">Шаг смещения окна пуллинга по вертикали (по строкам матрицы)</param>
        /// <param name="poolingType">Тип пуллинга. Максимальный или средний</param>
        /// <param name="padding">Тип заполнения краев</param>
        /// <param name="name"></param>
        public static Function Build(Variable input, int poolingWindowWidth, int poolingWindowHeight, int hStride, int vStride, PoolingType poolingType, Padding padding, string name)
        {
            var pooling = CNTKLib.Pooling(input, poolingType, new int[] { poolingWindowWidth, poolingWindowHeight }, new int[] { hStride, vStride }, new bool[] { padding == Padding.Valid });
            return CNTKLib.Alias(pooling, name);
        }
        public override Function Create(Function input, DeviceDescriptor device)
        {
            return Build(input, _poolingWindowWidth, _poolingWindowHeight, _hStride, _vStride, _poolingType, _padding, _name);
        }
        /// <summary>
        /// Создает пуллинг слой для двумерного вектора. Если предыдущий слой имеет не двумерный выход, выбрасывается исключение
        /// </summary>
        /// <param name="poolingWindowWidth">Ширина окна пуллинга</param>
        /// <param name="poolingWindowHeight">Высота окна пуллинга</param>
        /// <param name="hStride">Шаг смещения окна пуллинга по горизонтали (по столбцам матрицы)</param>
        /// <param name="vStride">Шаг смещения окна пуллинга по вертикали (по строкам матрицы)</param>
        /// <param name="poolingType">Тип пуллинга. Максимальный или средний</param>
        /// <param name="padding">Тип заполнения краев</param>
        /// <param name="name"></param>
        public Pooling2D(int poolingWindowWidth, int poolingWindowHeight, int hStride, int vStride, PoolingType poolingType, Padding padding = Padding.Valid, string name = "Pooling2D")
        {
            _poolingWindowWidth = poolingWindowWidth;
            _poolingWindowHeight = poolingWindowHeight;
            _hStride = hStride;
            _vStride = vStride;
            _poolingType = poolingType;
            _padding = padding;
            _name = name;
        }        
        public override string GetDescription()
        {
            return $"Pooling2D(W={_poolingWindowWidth}x{_poolingWindowHeight}S={_hStride}x{_vStride}T={_poolingType}P={_padding})";
        }
    }
}

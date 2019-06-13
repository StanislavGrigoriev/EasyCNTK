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
using System.Collections.Generic;
using System.Linq;

namespace EasyCNTK
{
    /// <summary>
    /// Реализует методы преобразования сырых данных в формат пригодный для обучения в CNTK
    /// </summary>
    public class ValueConverter
    {
        /// <summary>
        /// Разбивает входную последовательность на сегменты (подпоследовательности) равного размера
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="source">Исходная последовательность</param>
        /// <param name="segmentSize">Размер сегмента (количество элементов)</param>
        /// <returns></returns>
        protected IEnumerable<IList<T>> GetSegments<T>(IEnumerable<T> source, int segmentSize)
        {
            IList<T> list = new List<T>(segmentSize);
            foreach (var item in source)
            {
                list.Add(item);
                if (list.Count == segmentSize)
                {
                    yield return list;
                    list = new List<T>(segmentSize);
                }
            }
            if (list.Count > 0)
            {
                yield return list;
            }
        }
        /// <summary>
        /// Преобразует датасет в наборы обучающих примеров для использования в реккурентных сетях. 
        /// </summary>
        /// <typeparam name="T">Поддерживается <seealso cref="float"/>, <seealso cref="double"/></typeparam>
        /// <param name="features">Набор последовательностей (признаков). Каждая последовательность может быть переменной длинны, но одинаковой размерности (массивы из которых состоит последовательность, должны иметь одинаковую длину)</param>
        /// <param name="labels">Набор меток. Размерность меток должна быть одинаковая.</param>
        /// <param name="minibatchSize">Размер минипакета</param>
        /// <param name="device">Устройство для расчетов</param>
        /// <returns></returns>
        public IEnumerable<Minibatch> ConvertDatasetToMinibatch<T>(IEnumerable<IList<T[]>> features, IEnumerable<T[]> labels, int minibatchSize, DeviceDescriptor device) where T:IConvertible
        {
            var inputDimension = features.FirstOrDefault()[0].Length;
            var outputDimension = labels.FirstOrDefault().Length;
            var combined = features.Zip(labels, (f, l) => (f, l));
            foreach (var batchData in GetSegments(combined, minibatchSize))
            {
                //{}-miniBatch, ()-sequence, []-features.  
                //{ ([inputDim], [inputDim], [inputDim]), ([inputDim], [inputDim]) } => { [inputDim * 3], [inputDim * 2] }
                var featuresTransformed = batchData.Select(p => p.f.SelectMany(q => q));
                //{ [outputDim], [outputDim] } => { outputDim * 2 }
                var labelTransformed = batchData.SelectMany(p => p.l);

                Minibatch minibatch = new Minibatch();
                minibatch.Features = Value.CreateBatchOfSequences(new[] { inputDimension }, featuresTransformed, device);
                minibatch.Labels = Value.CreateBatch(new[] { outputDimension }, labelTransformed, device);
                minibatch.Size = batchData.Count;
                yield return minibatch;
            }
        }

        /// <summary>
        /// Преобразует датасет в наборы обучающих примеров для использования в CNTK. 
        /// </summary>
        /// <typeparam name="T">Поддерживается <seealso cref="float"/>, <seealso cref="double"/></typeparam>
        /// <param name="dataset">Датасет. Каждый пример должен содержать в начале массива признаки размерностью inputDim, а в конце метки классов размерностью outputDim. 
        /// Например inputDim = 3, outputDim = 2: [f1, f2, f3, l1, l2]</param>
        /// <param name="minibatchSize">Размер минипакета</param>
        /// <param name="inputDim">Размерность признаков (разрядность)</param>        
        /// <param name="device">Устройство для расчетов</param>
        /// <returns></returns>
        public IEnumerable<Minibatch> ConvertDatasetToMinibatch<T>(IEnumerable<IList<T>> dataset, int inputDim, int minibatchSize, DeviceDescriptor device) where T : IConvertible
        {
            var outputDim = (dataset.FirstOrDefault()?.Count ?? 0) - inputDim;
            foreach (var minibatchData in GetSegments(dataset, minibatchSize))
            {
                var features = minibatchData.SelectMany(p => p.Take(inputDim)).ToArray();
                var labels = minibatchData.SelectMany(p => p.Skip(inputDim)).ToArray();

                Minibatch minibatch = new Minibatch();
                minibatch.Size = minibatchData.Count;
                minibatch.Features = Value.CreateBatch(new int[] { inputDim }, features, device);
                minibatch.Labels = Value.CreateBatch(new int[] { outputDim }, labels, device);                

                yield return minibatch;
            }
        }
        /// <summary>
        /// Преобразует датасет в наборы обучающих примеров в 2Д для использования в CNTK. 
        /// </summary>
        /// <typeparam name="T">Поддерживается <seealso cref="float"/>, <seealso cref="double"/></typeparam>
        /// <param name="dataset">Датасет из 2Д примеров</param>
        /// <param name="minibatchSize">Размер минипакета</param>
        /// <param name="device">Устройство для расчетов</param>
        /// <returns></returns>
        public IEnumerable<Minibatch> ConvertDatasetToMinibatch<T>(IEnumerable<Sample2D<T>> dataset, int minibatchSize, DeviceDescriptor device) where T : IConvertible
        {
            foreach (var segment in GetSegments(dataset, minibatchSize))
            {
                var features = segment.SelectMany(p => getVector(p.Features));
                var labels = segment.SelectMany(p => p.Labels);

                Minibatch minibatch = new Minibatch();
                minibatch.Size = segment.Count;
                minibatch.Features = Value.CreateBatch(new int[] { segment[0].Features.GetLength(0), segment[0].Features.GetLength(1), 1 }, features, device);
                minibatch.Labels = Value.CreateBatch(new int[] { segment[0].Labels.GetLength(0) }, labels, device);   

                yield return minibatch;
            }
        }
        /// <summary>
        /// Преобразует нативный набор признаков в набор признаков в формате CNTK.
        /// </summary>
        /// <typeparam name="T">Поддерживается <seealso cref="float"/>, <seealso cref="double"/></typeparam>
        /// <param name="data">Набор признаков для каждого примера(sample)</param>
        /// <param name="minibatchSize">Размер пакета, по которым разбиваются признаки</param>
        /// <param name="device">Устройство для расчетов</param>
        /// <returns></returns>
        public IEnumerable<Value> ConvertDataToValue<T>(IEnumerable<IList<T>> data, int minibatchSize, DeviceDescriptor device) where T : IConvertible
        {
            int inputDim = data.FirstOrDefault()?.Count ?? 0;
            foreach (var segment in GetSegments(data, minibatchSize))
            {
                var features = segment.SelectMany(p => p).ToArray();
                var value = Value.CreateBatch(new int[] { inputDim }, features, device);
                yield return value;
            }
        }
        /// <summary>
        /// Преобразует нативный набор признаков (последовательность) в набор признаков в формате CNTK.
        /// </summary>
        /// <typeparam name="T">Поддерживается <seealso cref="float"/>, <seealso cref="double"/></typeparam>
        /// <param name="data">Набор признаков для каждого примера(sample), где пример - последовательность</param>
        /// <param name="minibatchSize">Размер пакета, по которым разбиваются признаки</param>
        /// <param name="device">Устройство для расчетов</param>
        /// <returns></returns>
        public IEnumerable<Value> ConvertDataToValue<T>(IEnumerable<IList<T[]>> data, int minibatchSize, DeviceDescriptor device) where T : IConvertible
        {
            int inputDim = data.FirstOrDefault()[0].Length;
            foreach (var segment in GetSegments(data, minibatchSize))
            {
                //{}-miniBatch, ()-sequence, []-features.  
                //{ ([inputDim], [inputDim], [inputDim]), ([inputDim], [inputDim]) } => { [inputDim * 3], [inputDim * 2] }
                var featuresTransformed = segment.Select(p => p.SelectMany(q => q));
                var value = Value.CreateBatchOfSequences(new[] { inputDim }, featuresTransformed, device);
                yield return value;
            }
        }
        /// <summary>
        /// Преобразует нативный набор признаков (2D) в набор признаков в формате CNTK.
        /// </summary>
        /// <typeparam name="T">Поддерживается <seealso cref="float"/>, <seealso cref="double"/></typeparam>
        /// <param name="data">Набор признаков для каждого примера(sample), где пример - 2D</param>
        /// <param name="minibatchSize">Размер пакета, по которым разбиваются признаки</param>
        /// <param name="device">Устройство для расчетов</param>
        /// <returns></returns>
        public IEnumerable<Value> ConvertDataToValue<T>(IEnumerable<Sample2D<T>> data, int minibatchSize, DeviceDescriptor device) where T : IConvertible
        {
            foreach (var segment in GetSegments(data, minibatchSize))
            {
                var features = segment.SelectMany(p => getVector(p.Features));
                var value = Value.CreateBatch(new int[] { segment[0].Features.GetLength(0), segment[0].Features.GetLength(1), 1 }, features, device);
                yield return value;
            }
        }

        private T[] getVector<T>(T[,] matrix)
        {
            var width = matrix.GetLength(0);
            var height = matrix.GetLength(1);
            var result = new T[width * height];
            for (int i = 0; i < width; i++)
            {
                for (int j = 0; j < height; j++)
                {
                    result[(i + 1) * j] = matrix[i, j];
                }
            }
            return result;
        }
    }
    /// <summary>
    /// Представляет пример, признаки у которого представлены в 2Д (матрица)
    /// </summary>
    /// <typeparam name="T">Поддерживается <seealso cref="float"/>, <seealso cref="double"/></typeparam>
    public class Sample2D<T> where T : IConvertible
    {
        public T[,] Features { get; set; }
        public T[] Labels { get; set; }
    }

    /// <summary>
    /// Представляет пачку данных для обучения (небольшой набор обучающих примеров)
    /// </summary>
    public class Minibatch
    {
        /// <summary>
        /// Размер пачки (количество обучающих примеров в пачке)
        /// </summary>
        public int Size { get; set; }
        /// <summary>
        /// Признаки
        /// </summary>
        public Value Features { get; set; }
        /// <summary>
        /// Метки классов/непрерывные значения меток
        /// </summary>
        public Value Labels { get; set; }
    }
}

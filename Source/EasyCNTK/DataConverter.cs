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
    /// Реализует методы преобразования нативных данных в формат пригодный для обучения в CNTK
    /// </summary>
    public class DataConverter:IDisposable
    {
        protected DeviceDescriptor Device { get; set; }
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
        protected T[] MatrixToVector<T>(T[,] matrix)
        {
            var rows = matrix.GetLength(0);
            var columns = matrix.GetLength(1);
            var result = new T[rows * columns];
            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < columns; j++)
                {
                    result[(i + 1) * j] = matrix[i, j];
                }
            }
            return result;
        }
        protected int GetRowsCount<T>(T[,] matrix)
        {
            return matrix.GetLength(0);
        }
        protected int GetColumnsCount<T>(T[,] matrix)
        {
            return matrix.GetLength(1);
        }
        /// <summary>
        /// Инициализирует конвертер для работы с указанным устройством (CPU, GPU)
        /// </summary>
        /// <param name="device">Устройство для расчетов</param>
        public DataConverter(DeviceDescriptor device)
        {            
            Device = device ?? throw new ArgumentNullException(nameof(device));
        }
        /// <summary>
        /// Преобразует датасет в наборы обучающих примеров для использования в реккурентных сетях. 
        /// </summary>
        /// <typeparam name="T">Поддерживается <seealso cref="float"/>, <seealso cref="double"/></typeparam>
        /// <param name="features">Набор последовательностей (признаков). Каждая последовательность может быть переменной длинны, но одинаковой размерности (массивы из которых состоит последовательность, должны иметь одинаковую длину)</param>
        /// <param name="labels">Набор меток. Размерность меток должна быть одинаковая.</param>
        /// <param name="minibatchSize">Размер минипакета</param>
        /// <returns></returns>
        public IEnumerable<Minibatch> ConvertDatasetToMinibatch<T>(IEnumerable<IList<T[]>> features, IEnumerable<T[]> labels, int minibatchSize) where T:IConvertible
        {
            var inputDimension = features.FirstOrDefault()?[0].Length ?? 0;
            var outputDimension = labels.FirstOrDefault()?.Length ?? 0;
            var combined = features.Zip(labels, (f, l) => (f, l));
            foreach (var batchData in GetSegments(combined, minibatchSize))
            {
                //{}-miniBatch, ()-sequence, []-features.  
                //{ ([inputDim], [inputDim], [inputDim]), ([inputDim], [inputDim]) } => { [inputDim * 3], [inputDim * 2] }
                var featuresTransformed = batchData.Select(p => p.f.SelectMany(q => q));
                //{ [outputDim], [outputDim] } => { outputDim * 2 }
                var labelTransformed = batchData.SelectMany(p => p.l);

                Minibatch minibatch = new Minibatch();
                minibatch.Features = Value.CreateBatchOfSequences(new[] { inputDimension }, featuresTransformed, Device);
                minibatch.Labels = Value.CreateBatch(new[] { outputDimension }, labelTransformed, Device);
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
        /// <returns></returns>
        public IEnumerable<Minibatch> ConvertDatasetToMinibatch<T>(IEnumerable<T[]> dataset, int inputDim, int minibatchSize) where T : IConvertible
        {
            var outputDim = (dataset.FirstOrDefault()?.Length ?? 0) - inputDim;
            foreach (var minibatchData in GetSegments(dataset, minibatchSize))
            {
                var features = minibatchData.SelectMany(p => p.Take(inputDim)).ToArray();
                var labels = minibatchData.SelectMany(p => p.Skip(inputDim)).ToArray();

                Minibatch minibatch = new Minibatch();
                minibatch.Size = minibatchData.Count;
                minibatch.Features = Value.CreateBatch(new int[] { inputDim }, features, Device);
                minibatch.Labels = Value.CreateBatch(new int[] { outputDim }, labels, Device);                

                yield return minibatch;
            }
        }
        /// <summary>
        /// Преобразует датасет в наборы обучающих примеров в 2D для использования в CNTK. 
        /// </summary>
        /// <typeparam name="T">Поддерживается <seealso cref="float"/>, <seealso cref="double"/></typeparam>
        /// <param name="features">Набор признаков в 2D</param>
        /// <param name="labels">Набор меток. Размерность меток должна быть одинаковая.</param>
        /// <param name="minibatchSize">Размер минипакета</param>
        /// <returns></returns>
        public IEnumerable<Minibatch> ConvertDatasetToMinibatch<T>(IEnumerable<T[,]> features, IEnumerable<T[]> labels, int minibatchSize) where T : IConvertible
        {
            var combined = features.Zip(labels, (f, l) => (f, l));
            foreach (var segment in GetSegments(combined, minibatchSize))
            {
                var featuresData = segment.SelectMany(p => MatrixToVector(p.f));
                var labelsData = segment.SelectMany(p => p.l);

                Minibatch minibatch = new Minibatch();
                minibatch.Size = segment.Count;
                minibatch.Features = Value.CreateBatch(new int[] { GetRowsCount(segment[0].f), GetColumnsCount(segment[0].f), 1 }, featuresData, Device);
                minibatch.Labels = Value.CreateBatch(new int[] { segment[0].l.Length }, labelsData, Device);   

                yield return minibatch;
            }
        }
        /// <summary>
        /// Преобразует нативный набор признаков в набор признаков в формате CNTK.
        /// </summary>
        /// <typeparam name="T">Поддерживается <seealso cref="float"/>, <seealso cref="double"/></typeparam>
        /// <param name="data">Набор признаков для каждого примера(sample)</param>
        /// <param name="minibatchSize">Размер пакета, по которым разбиваются признаки</param>
        /// <returns></returns>
        public IEnumerable<Value> ConvertDataToValue<T>(IEnumerable<T[]> data, int minibatchSize) where T : IConvertible
        {
            int inputDim = data.FirstOrDefault()?.Length ?? 0;
            foreach (var segment in GetSegments(data, minibatchSize))
            {
                var features = segment.SelectMany(p => p).ToArray();
                var value = Value.CreateBatch(new int[] { inputDim }, features, Device);
                yield return value;
            }
        }
        /// <summary>
        /// Преобразует нативный набор признаков (последовательность) в набор признаков в формате CNTK.
        /// </summary>
        /// <typeparam name="T">Поддерживается <seealso cref="float"/>, <seealso cref="double"/></typeparam>
        /// <param name="data">Набор признаков для каждого примера(sample), где пример - последовательность</param>
        /// <param name="minibatchSize">Размер пакета, по которым разбиваются признаки</param>
        /// <returns></returns>
        public IEnumerable<Value> ConvertDataToValue<T>(IEnumerable<IList<T[]>> data, int minibatchSize) where T : IConvertible
        {
            int inputDim = data.FirstOrDefault()[0].Length;
            foreach (var segment in GetSegments(data, minibatchSize))
            {
                //{}-miniBatch, ()-sequence, []-features.  
                //{ ([inputDim], [inputDim], [inputDim]), ([inputDim], [inputDim]) } => { [inputDim * 3], [inputDim * 2] }
                var featuresTransformed = segment.Select(p => p.SelectMany(q => q));
                var value = Value.CreateBatchOfSequences(new[] { inputDim }, featuresTransformed, Device);
                yield return value;
            }
        }
        /// <summary>
        /// Преобразует нативный набор признаков (2D) в набор признаков в формате CNTK.
        /// </summary>
        /// <typeparam name="T">Поддерживается <seealso cref="float"/>, <seealso cref="double"/></typeparam>
        /// <param name="data">Набор признаков для каждого примера(sample), где пример - 2D</param>
        /// <param name="minibatchSize">Размер пакета, по которым разбиваются признаки</param>
        /// <returns></returns>
        public IEnumerable<Value> ConvertDataToValue<T>(IEnumerable<T[,]> data, int minibatchSize) where T : IConvertible
        {
            foreach (var segment in GetSegments(data, minibatchSize))
            {
                var features = segment.SelectMany(p => MatrixToVector(p));
                var value = Value.CreateBatch(new int[] { GetRowsCount(segment[0]), GetColumnsCount(segment[0]), 1 }, features, Device);
                yield return value;
            }
        }
        /// <summary>
        /// Преобразует датасет в наборы обучающих примеров для использования в CNTK. Используется для обучения моделей с несколькими выходами.
        /// </summary>
        /// <typeparam name="T">Поддерживается <seealso cref="float"/>, <seealso cref="double"/></typeparam>
        /// <param name="features">Набор признаков для каждого примера(sample)</param>
        /// <param name="labels">Набор меток для каждого выхода модели, размерность для каждого выхода может быть своя. </param>
        /// <param name="minibatchSize">Размер минипакета</param>
        /// <returns></returns>
        public IEnumerable<MinibatchMultiOutput> ConvertDatasetToMinibatchMultiOutput<T>(IEnumerable<T[]> features, IEnumerable<T[][]> labels, int minibatchSize)
        {
            int inputDim = features.FirstOrDefault()?.Length ?? 0;
            int outputCount = labels.FirstOrDefault()?.Length ?? 0;
            var combined = features.Zip(labels, (f, l) => (f, l));
            foreach (var segment in GetSegments(combined, minibatchSize))
            {
                var featuresData = segment.SelectMany(p => p.f);
                var labelsData = new T[outputCount][];
                for (int i = 0; i < outputCount; i++)
                {
                    labelsData[i] = segment.SelectMany(p => p.l[i]).ToArray();
                }                

                MinibatchMultiOutput minibatch = new MinibatchMultiOutput();
                minibatch.Size = segment.Count;
                minibatch.Features = Value.CreateBatch(new int[] { inputDim }, featuresData, Device);
                minibatch.Labels = labelsData
                    .Select(label => Value.CreateBatch(new int[] { label.Length / segment.Count }, label, Device))
                    .ToArray();

                yield return minibatch;
            }
        }
        /// <summary>
        /// Преобразует датасет в наборы обучающих примеров для использования в CNTK. Используется для обучения реккурентных моделей с несколькими выходами.
        /// </summary>
        /// <typeparam name="T">Поддерживается <seealso cref="float"/>, <seealso cref="double"/></typeparam>
        /// <param name="features">Набор признаков для каждого примера(sample), при пример - последовательность</param>
        /// <param name="labels">Набор меток для каждого выхода модели, размерность для каждого выхода может быть своя. </param>
        /// <param name="minibatchSize">Размер минипакета</param>
        /// <returns></returns>
        public IEnumerable<MinibatchMultiOutput> ConvertDatasetToMinibatchMultiOutput<T>(IEnumerable<IList<T[]>> features, IEnumerable<T[][]> labels, int minibatchSize)
        {
            int inputDim = features.FirstOrDefault()?[0].Length ?? 0;
            int outputCount = labels.FirstOrDefault()?.Length ?? 0;
            var combined = features.Zip(labels, (f, l) => (f, l));
            foreach (var segment in GetSegments(combined, minibatchSize))
            {
                var featuresData = segment.Select(p => p.f.SelectMany(q => q));
                var labelsData = new T[outputCount][];
                for (int i = 0; i < outputCount; i++)
                {
                    labelsData[i] = segment.SelectMany(p => p.l[i]).ToArray();
                }

                MinibatchMultiOutput minibatch = new MinibatchMultiOutput();
                minibatch.Size = segment.Count;
                minibatch.Features = Value.CreateBatch(new int[] { inputDim }, featuresData, Device);
                minibatch.Labels = labelsData
                    .Select(label => Value.CreateBatch(new int[] { label.Length / segment.Count }, label, Device))
                    .ToArray();

                yield return minibatch;
            }
        }
        /// <summary>
        /// Преобразует 2D датасет в наборы обучающих примеров для использования в CNTK. Используется для обучения моделей с несколькими выходами.
        /// </summary>
        /// <typeparam name="T">Поддерживается <seealso cref="float"/>, <seealso cref="double"/></typeparam>
        /// <param name="features">Набор признаков для каждого примера(sample) в 2D</param>
        /// <param name="labels">Набор меток для каждого выхода модели, размерность для каждого выхода может быть своя. </param>
        /// <param name="minibatchSize">Размер минипакета</param>
        /// <returns></returns>
        public IEnumerable<MinibatchMultiOutput> ConvertDatasetToMinibatchMultiOutput<T>(IEnumerable<T[,]> features, IEnumerable<T[][]> labels, int minibatchSize) where T:IConvertible
        {            
            int outputCount = labels.FirstOrDefault()?.Length ?? 0;
            var combined = features.Zip(labels, (f, l) => (f, l));
            foreach (var segment in GetSegments(combined, minibatchSize))
            {
                var featuresData = segment.SelectMany(p => MatrixToVector(p.f));
                var labelsData = new T[outputCount][];
                for (int i = 0; i < outputCount; i++)
                {
                    labelsData[i] = segment.SelectMany(p => p.l[i]).ToArray();
                }

                MinibatchMultiOutput minibatch = new MinibatchMultiOutput();
                minibatch.Size = segment.Count;
                minibatch.Features = Value.CreateBatch(new int[] { GetRowsCount(segment[0].f), GetColumnsCount(segment[0].f), 1 }, featuresData, Device);
                minibatch.Labels = labelsData
                    .Select(label => Value.CreateBatch(new int[] { label.Length / segment.Count }, label, Device))
                    .ToArray();

                yield return minibatch;
            }
        }


        #region IDisposable Support
        private bool disposedValue = false; // Для определения избыточных вызовов

        protected virtual void Dispose(bool disposing)
        {
            if (!disposedValue)
            {
                if (disposing)
                {
                    // TODO: освободить управляемое состояние (управляемые объекты).
                    Device.Dispose();
                }

                // TODO: освободить неуправляемые ресурсы (неуправляемые объекты) и переопределить ниже метод завершения.
                // TODO: задать большим полям значение NULL.
                Device = null;
                disposedValue = true;
            }
        }

        // TODO: переопределить метод завершения, только если Dispose(bool disposing) выше включает код для освобождения неуправляемых ресурсов.
        ~DataConverter()
        {
            // Не изменяйте этот код. Разместите код очистки выше, в методе Dispose(bool disposing).
            Dispose(false);
        }

        // Этот код добавлен для правильной реализации шаблона высвобождаемого класса.
        public void Dispose()
        {
            // Не изменяйте этот код. Разместите код очистки выше, в методе Dispose(bool disposing).
            Dispose(true);
            // TODO: раскомментировать следующую строку, если метод завершения переопределен выше.
            GC.SuppressFinalize(this);
        }
        #endregion
    }

    /// <summary>
    /// Представляет пачку данных для обучения
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
    /// <summary>
    /// Представляет пачку данных для обучения моделей с несколькими выходами
    /// </summary>
    public class MinibatchMultiOutput
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
        /// Метки классов/непрерывные значения меток, для каждого выхода модели
        /// </summary>
        public Value[] Labels { get; set; }
    }
}

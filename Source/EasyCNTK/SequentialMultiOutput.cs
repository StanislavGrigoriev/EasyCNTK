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
using EasyCNTK.Layers;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;

namespace EasyCNTK
{
    /// <summary>
    /// Реализует операции конструирования модели прямого распространения c одним входом и несколькими выходами
    /// </summary>
    /// <typeparam name="T">Тип данных. Поддерживается <seealso cref="float"/>, <seealso cref="double"/></typeparam>
    public sealed class SequentialMultiOutput<T>: IDisposable where T:IConvertible
    {
        class Branch
        {
            public int Index { get; set; }
            public string Name { get; set; }
            public string ArchitectureDescription { get; set; }
            public Function Model { get; set; }
        }

        public const string PREFIX_FILENAME_DESCRIPTION = "ArchitectureDescription";
        private DeviceDescriptor _device;
        private Function _model;
        private string _architectureDescription;
        private Dictionary<string, Branch> _branches = new Dictionary<string, Branch>();        
        private bool _isCompiled = false;
        private string getArchitectureDescription()
        {
            var descriptionBranches = _branches
                .Values
                .OrderBy(p => p.Index)
                .Select(p => p.ArchitectureDescription);
            StringBuilder stringBuilder = new StringBuilder(_architectureDescription);
            foreach (var branch in descriptionBranches)
            {
                stringBuilder.Append(branch);
                stringBuilder.Append("[OUT]");
            }
            return stringBuilder.ToString();
        }

        /// <summary>
        /// Загружает модель из файла. Так же пытается прочитать описание архитектуры сети: 
        /// 1) Из файла ArchitectureDescription{имя_файла_модели}.txt 
        /// 2) Из имени файла модели ориентируясь на наличение [IN] и [OUT] тегов. Если это не удается, то описание конфигурации: Unknown.
        /// </summary>
        /// <typeparam name="T">Тип данных. Поддерживается <seealso cref="float"/>, <seealso cref="double"/></typeparam>
        /// <param name="device">Устройство для загрузки</param>
        /// <param name="filePath">Путь к файлу модели</param>
        /// <param name="modelFormat">Формат модели</param>
        /// <returns></returns>
        public static SequentialMultiOutput<T> LoadModel(DeviceDescriptor device, string filePath, ModelFormat modelFormat = ModelFormat.CNTKv2)
        {
            return new SequentialMultiOutput<T>(device, filePath, modelFormat);
        }
        /// <summary>
        /// Инициализирeует нейросеть с размерностью входного вектора без слоев
        /// </summary>
        /// <param name="inputShape">Тензор, описывающий форму входа нейросети (входных данных)</param>
        /// <param name="device">Устройство на котором создается сеть</param>
        /// <param name="outputIsSequence">Указывает, что выход сети - последовательность.</param>
        /// <param name="inputName">Имя входа нейросети</param>
        /// <param name="isSparce">Указывает, что вход это вектор One-Hot-Encoding и следует использовать внутреннюю оптимизацию CNTK для увеличения производительности.</param>
        public SequentialMultiOutput(DeviceDescriptor device, int[] inputShape, bool outputIsSequence = false, string inputName = "Input", bool isSparce = false)
        {
            _device = device;
            var dataType = typeof(T) == typeof(double) ? DataType.Double : DataType.Float;
            _model = outputIsSequence
                ? Variable.InputVariable(inputShape, dataType, inputName, new[] { Axis.DefaultBatchAxis() }, isSparce)
                : Variable.InputVariable(inputShape, dataType, inputName, null, isSparce);
            var shape = "";
            inputShape.ToList().ForEach(p =>
            {
                shape += p.ToString() + "x";
            });
            shape = shape.Substring(0, shape.Length - 1);
            _architectureDescription = $"[IN]{shape}";
        }

        private SequentialMultiOutput(DeviceDescriptor device, string filePath, ModelFormat modelFormat = ModelFormat.CNTKv2)
        {
            _isCompiled = true;
            _device = device;
            _model = Function.Load(filePath, device, modelFormat);
            var dataType = typeof(T) == typeof(double) ? DataType.Double : DataType.Float;
            if (_model.Output.DataType != dataType)
            {
                throw new ArgumentException($"Универсальный параметр {nameof(T)} не сответствует типу данных в модели. Требуемый тип: {_model.Output.DataType}");
            }
            try
            {
                _architectureDescription = "Unknown";

                var pathToFolder = Directory.GetParent(filePath).FullName;
                var descriptionPath = Path.Combine(pathToFolder, $"{PREFIX_FILENAME_DESCRIPTION}_{filePath}.txt");
                if (File.Exists(descriptionPath))
                {
                    var description = File.ReadAllText(descriptionPath);
                    var index = description.IndexOf("[OUT]");
                    _architectureDescription = index != -1 ? description.Remove(index) : description;
                    return;
                }

                var fileName = Path.GetFileName(filePath);
                var indexIn = fileName.IndexOf("[IN]");
                var indexOut = fileName.LastIndexOf("[OUT]");
                bool fileNameContainsArchitectureDescription = indexIn != -1 && indexOut != -1 && indexIn < indexOut;
                if (fileNameContainsArchitectureDescription)
                {
                    _architectureDescription = fileName.Substring(indexIn, indexOut - indexIn);
                }
            }
            catch (Exception ex)
            {
                System.Diagnostics.Debug.WriteLine($"Ошибка загрузки файла конфигурации модели. {ex.Message}");
            }
        }
        /// <summary>
        /// Добавляет заданный слой (стыкует к последнему добавленному слою)
        /// </summary>
        /// <param name="layer">Слой для стыковки</param>
        public void Add(Layer layer)
        {
            if (_isCompiled)
                throw new NotSupportedException("Изменение скомпилированной модели не поддерживается.");
            _model = layer.Create(_model, _device);
            _architectureDescription += $"-{layer.GetDescription()}";
        }
        /// <summary>
        /// Добавляет заданный слой в указанную ветвь (стыкует к последнему добавленному слою ветви)
        /// </summary>
        /// <param name="branch">Имя ветви. Должно совпадать с одним из имен указанных при вызове <seealso cref="SplitToBranches(string[])"/></param>
        /// <param name="layer">Слой для стыковки</param>
        public void AddToBranch(string branch, Layer layer)
        {
            if (_isCompiled)
                throw new NotSupportedException("Изменение скомпилированной модели не поддерживается.");
            if (_branches.Count == 0)
                throw new NotSupportedException("Добавления слоя к ветви без предварительного создания ветвей не поддерживается, сначала создайте ветви методом SplitToBranches()");
            if (_branches.TryGetValue(branch, out var branchOutput))
            {
                _branches[branch].Model = layer.Create(branchOutput.Model, _device);
                _branches[branch].ArchitectureDescription += $"-{layer.GetDescription()}";                
            }
            else
            {
                throw new ArgumentException($"Ветви с именем '{branch}' не существует.", nameof(branch));
            }
        }
        /// <summary>
        /// Разбивает основную последовательность слоев на несколько ветвей
        /// </summary>
        /// <param name="branchNames">Названия ветвей, каждой ветви в порядке перечисления будет сопоставлен соответсвующий выход сети. Названия должны быть уникальны.</param>
        public void SplitToBranches(params string[] branchNames)
        {
            if (_isCompiled)
                throw new NotSupportedException("Изменение скомпилированной модели не поддерживается.");
            if (_branches.Count != 0)
                throw new NotSupportedException("Повторное разбиение сущеcтвующих ветвей на новые ветви не поддерживается.");
            if (branchNames.Length < 2)
                throw new NotSupportedException("Разбиение возможно минимум на 2 ветви.");
            _branches = branchNames
                .Select((branch, index) => (branch, index, _model))
                .ToDictionary(p => p.branch, q => new Branch()
                {
                    Index = q.index,
                    Name = q.branch,
                    ArchitectureDescription = $"-#{q.branch}",
                    Model = _model
                });                      
        }
        /// <summary>
        /// Компилирует все созданные ветви в одну модель
        /// </summary>
        public void Compile()
        {
            var outputs = _branches
                .Values
                .OrderBy(p => p.Index)
                .Select(p => (Variable)p.Model)
                .ToList();
            _model = CNTKLib.Combine(new VariableVector(outputs));
            
            _isCompiled = true;
        }

        //private bool _isDisposed = false;
        //public void Dispose()
        //{
        //    if (!_isDisposed)
        //    {
        //        _device.Dispose();
        //        _model.Dispose();
        //        foreach (var item in _branches.Values)
        //        {
        //            item.Model.Dispose();
        //        }
        //    }
        //    _isDisposed = true;
        //}
        /// <summary>
        /// Скомпилированная модель CNTK 
        /// </summary>
        public Function Model
        {
            get
            {
                if (!_isCompiled)
                    throw new NotSupportedException("Использование нескомпилированной модели не поддерживается. Скомпилируйте вызвав Compile()");
                return _model;
            }
        }
        public override string ToString()
        {
            return getArchitectureDescription();
        }
        /// <summary>
        /// Сохраняет модель в файл.
        /// </summary>
        /// <param name="filePath">Путь для сохранения модели (включая имя файла и расширение)</param>
        /// <param name="saveArchitectureDescription">Указывает, следует ли сохранить описание архитектуры в отдельном файле: ArchitectureDescription_{имя-файла-модели}.txt</param>
        public void SaveModel(string filePath, bool saveArchitectureDescription = true)
        {
            Model.Save(filePath);
            if (saveArchitectureDescription)
            {
                var fileName = Path.GetFileName(filePath);
                var pathToFolder = Directory.GetParent(filePath).FullName;
                var descriptionPath = Path.Combine(pathToFolder, $"{PREFIX_FILENAME_DESCRIPTION}_{fileName}.txt");
                using (var stream = File.CreateText(descriptionPath))
                {
                    stream.Write(getArchitectureDescription());
                }
            }
        }

        #region IDisposable Support
        private bool disposedValue = false; // Для определения избыточных вызовов

        void Dispose(bool disposing)
        {
            if (!disposedValue)
            {
                if (disposing)
                {
                    // TODO: освободить управляемое состояние (управляемые объекты).
                    _device.Dispose();
                    _model.Dispose();
                    foreach (var item in _branches.Values)
                    {
                        item.Model.Dispose();
                    }
                }

                // TODO: освободить неуправляемые ресурсы (неуправляемые объекты) и переопределить ниже метод завершения.
                // TODO: задать большим полям значение NULL.
                _device = null;
                _model = null;
                _branches = null;

                disposedValue = true;
            }
        }

        // TODO: переопределить метод завершения, только если Dispose(bool disposing) выше включает код для освобождения неуправляемых ресурсов.
        ~SequentialMultiOutput()
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
}



using CNTK;
using EasyCNTK.Layers;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace EasyCNTK
{
    /// <summary>
    /// Реализует операции конструирования модели прямого распространения
    /// </summary>
    /// <typeparam name="T">Тип данных. Поддерживается <seealso cref="float"/>, <seealso cref="double"/></typeparam>
    public sealed class Sequential<T>
    {
        public const string PREFIX_FILENAME_DESCRIPTION = "ConfigurationDescription";        
        private DeviceDescriptor _device { get; }
        private Function _model { get; set; }
        private string _configurationDescription { get; set; }
        private Dictionary<string, Function> _shortcutConnectionInputs = new Dictionary<string, Function>();   
     
        private string getDescription()
        {
            var shortcuts = _shortcutConnectionInputs.Keys.ToList();
            foreach (var shortcut in shortcuts)
            {
                if (!_configurationDescription.Contains($"ShortOut({shortcut})"))
                {
                    _configurationDescription = _configurationDescription.Replace($"-ShortIn({shortcut})", "");
                }
            }
            return _configurationDescription;
        }
        /// <summary>
        /// Загружает модель из файла. Так же пытается загрузить описание архитектуры сети из файла ConfigurationDescription_{имя_файла_модели}.txt, если это не удается, то описание конфигурации остается пустым.
        /// </summary>
        /// <typeparam name="T">Тип данных. Поддерживается <seealso cref="float"/>, <seealso cref="double"/></typeparam>
        /// <param name="device">Устройство для загрузки</param>
        /// <param name="filePath">Путь к файлу модели</param>
        /// <param name="modelFormat">Формат модели</param>
        /// <returns></returns>
        public static Sequential<T> LoadModel(DeviceDescriptor device, string filePath, ModelFormat modelFormat = ModelFormat.CNTKv2)
        {
            return new Sequential<T>(device, filePath, modelFormat);
        }

        /// <summary>
        /// Инициализирeует нейросеть с размерностью входного вектора без слоев
        /// </summary>
        /// <param name="inputShape">Тензор, описывающий форму входа нейросети (входных данных)</param>
        /// <param name="device">Устройство на котором создается сеть</param>
        /// <param name="inputDynamicAxes">Список динамических осей. Добавьте ось <seealso cref="Axis.DefaultBatchAxis()"/>, если выход вашей сети - последовательность.</param>
        /// <param name="isSparce">Указывает, что вход это вектор One-Hot-Encoding и следует использовать внутреннюю оптимизацию CNTK для увеличения производительности.</param>
        public Sequential(DeviceDescriptor device, int[] inputShape, IList<Axis> inputDynamicAxes = null, bool isSparce = false)
        {
            _device = device;
            var dataType = typeof(T) == typeof(double) ? DataType.Double : DataType.Float;
            _model = Variable.InputVariable(inputShape, dataType, "Input", inputDynamicAxes, isSparce);
            var shape = "";
            inputShape.ToList().ForEach(p =>
            {
                shape += p.ToString() + "x";
            });
            shape = shape.Substring(0, shape.Length - 1);
            _configurationDescription = $"INPUT{shape}";
        }

         private Sequential(DeviceDescriptor device, string filePath, ModelFormat modelFormat = ModelFormat.CNTKv2)
        {
            _device = device;
            _model = Function.Load(filePath, device, modelFormat);
            var dataType = typeof(T) == typeof(double) ? DataType.Double : DataType.Float;
            if (_model.Output.DataType != dataType)
            {
                throw new ArgumentException($"Универсальный параметр TElement не сответствует типу данных в модели. Требуемый тип: {_model.Output.DataType}");
            }
            try
            {
                var pathToFolder = Directory.GetParent(filePath).FullName;
                var descriptionPath = Path.Combine(pathToFolder, $"{PREFIX_FILENAME_DESCRIPTION}_{filePath}.txt");
                if (File.Exists(descriptionPath))
                {
                    _configurationDescription = File.ReadAllText(descriptionPath);
                }
                else
                {
                    _configurationDescription = "";
                }                
            }
            catch (Exception ex)
            {
                System.Diagnostics.Debug.WriteLine($"Ошибка загрузки файла конфигурации модели. {ex.Message}");
                _configurationDescription = "";
            }
        }
        /// <summary>
        /// Добавляет заданный слой (стыкует к последнему добавленному слою)
        /// </summary>
        /// <param name="layer">Слой для стыковки</param>
        public void Add(Layer layer)
        {
            _model = layer.Create(_model, _device);
            _configurationDescription += $"-{layer.GetDescription()}";
        }
        /// <summary>
        /// Создает входную точку для SC, из которой можно создать соединение к следующим слоям сети. Для одной входной точки должна существовать как минимум одна выходная точка, иначе соединение игнорируется в модели.
        /// </summary>
        /// <param name="nameShortcutConnection">Название точки входа, из которой будет пробрасываться соединение. В рамках сети должно быть уникальным</param>
        public void CreateInputPointForShortcutConnection(string nameShortcutConnection)
        {
            _shortcutConnectionInputs.Add(nameShortcutConnection, _model);
            _configurationDescription += $"-ShortIn({nameShortcutConnection})";
        }
        /// <summary>
        /// Создает выходную точку для SC, к которой пробрасывается соединение из ранее созданной входной точки. Для одной входной точки может существовать несколько выходных точек.
        /// </summary>
        /// <param name="nameShortcutConnection">Название точки входа, из которой пробрасывается соединение.</param>
        public void CreateOutputPointForShortcutConnection(string nameShortcutConnection)
        {
            if (_shortcutConnectionInputs.TryGetValue(nameShortcutConnection, out var input))
            {
                if (input.Output.Shape.Equals(_model.Output.Shape))
                {
                     _model = CNTKLib.Plus(_model, input);                    
                }
                else if (input.Output.Shape.Rank != 1 && _model.Output.Shape.Rank == 1) // [3x4x2] => [5]
                {
                    int targetDim = _model.Output.Shape[0];
                    int inputDim = input.Output.Shape.Dimensions.Aggregate((d1, d2) => d1 * d2);
                    var inputVector = CNTKLib.Reshape(input, new[] { inputDim });

                    var scale = new Parameter(new[] { targetDim, inputDim }, input.Output.DataType, CNTKLib.UniformInitializer(CNTKLib.DefaultParamInitScale), _device);
                    var scaled = CNTKLib.Times(scale, inputVector);

                    var reshaped = CNTKLib.Reshape(scaled, _model.Output.Shape);
                    _model = CNTKLib.Plus(reshaped, _model);
                }
                else if (input.Output.Shape.Rank == 1 && _model.Output.Shape.Rank != 1) // [5] => [3x4x2]
                {
                    int targetDim = _model.Output.Shape.Dimensions.Aggregate((d1, d2) => d1 * d2);
                    var inputDim = input.Output.Shape[0];
                    var scale = new Parameter(new[] { targetDim, inputDim}, input.Output.DataType, CNTKLib.UniformInitializer(CNTKLib.DefaultParamInitScale), _device);
                    var scaled = CNTKLib.Times(scale, input);

                    var reshaped = CNTKLib.Reshape(scaled, _model.Output.Shape);
                    _model = CNTKLib.Plus(reshaped, _model);
                }
                else // [3x4x2] => [4x5x1] || [3x1] => [5x7x8x1]
                {
                    var inputDim = input.Output.Shape.Dimensions.Aggregate((d1, d2) => d1 * d2);
                    var inputVector = CNTKLib.Reshape(input, new[] { inputDim });

                    var targetDim = _model.Output.Shape.Dimensions.Aggregate((d1, d2) => d1 * d2);
                    var scale = new Parameter(new[] { targetDim, inputDim }, input.Output.DataType, CNTKLib.UniformInitializer(CNTKLib.DefaultParamInitScale), _device);
                    var scaled = CNTKLib.Times(scale, inputVector);
                    
                    var reshaped = CNTKLib.Reshape(scaled, _model.Output.Shape);
                    _model = CNTKLib.Plus(reshaped, _model);
                }
                _configurationDescription += $"-ShortOut({nameShortcutConnection})";
            }
        }
        /// <summary>
        /// Сконфигурированная модель CNTK
        /// </summary>
        public Function Model { get => _model; }
        /// <summary>
        /// Сохраняет модель в файл. При сохранении дополнительно создается текстовый файл с описанием архитектуры сети: ConfigurationDescription_{имя_файла_модели}.txt
        /// </summary>
        /// <param name="filePath">Путь для сохранения модели (включая имя файла и расширение)</param>
        public void SaveModel(string filePath)
        {
            _model.Save(filePath);
            var fileName = Path.GetFileName(filePath);
            var pathToFolder = Directory.GetParent(filePath).FullName;
            var descriptionPath = Path.Combine(pathToFolder, $"{PREFIX_FILENAME_DESCRIPTION}_{fileName}.txt");
            using (var stream = File.CreateText(descriptionPath))
            {
                stream.WriteLine(getDescription());
            }
        }       
        public override string ToString()
        {
            return getDescription();
        }
    }
}

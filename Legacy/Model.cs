using CNTK;
using System;
using System.Collections.Generic;
using System.Linq;
using EasyCNTK.ActivationFunctions;
using EasyCNTK.Layers;

namespace EasyCNTK.Legacy
{

    /// <summary>
    /// Полносвязная нейросеть прямого распространения
    /// </summary>
    [Obsolete]
    public class Model<T>
    {
        private Function _net;
        private Variable _inputVariable;
        private DeviceDescriptor _device;
        private DataType _dataType;

        /// <summary>
        /// Создает полносвязный слой с заданной функцией активации
        /// </summary>
        /// <param name="input">Входная переменная(слой) заданной разрядности</param>
        /// <param name="outputDim">Выходная разрядность(кол-во нейронов)</param>
        /// <param name="activationFunction">Функция активации</param>
        /// <param name="device">Устройство на котором производится расчет</param>
        /// <param name="name">Имя слоя</param>
        /// <returns></returns>
        private Function createFullyConnectedLinearLayer(Variable input, int outputDim, ActivationFunction activationFunction, DeviceDescriptor device, string name)
        {
            if (input.Shape.Rank != 1)
            {
                // если данные не одномерные разворачиваем входной тензор в вектор
                int newDim = input.Shape.Dimensions.Aggregate((d1, d2) => d1 * d2);
                input = CNTKLib.Reshape(input, new int[] { newDim });
            }

            var inputDim = input.Shape[0];
            var weight = new Parameter(new int[] { outputDim, inputDim }, _dataType, CNTKLib.GlorotUniformInitializer(
                CNTKLib.DefaultParamInitScale,
                    CNTKLib.SentinelValueForInferParamInitRank,
                    CNTKLib.SentinelValueForInferParamInitRank, 1), device);
            var bias = new Parameter(new int[] { outputDim }, _dataType, 0, device);
            var fullyConnected = CNTKLib.Times(weight, input) + bias;

            if (activationFunction == null)
            {
                return Function.Alias(fullyConnected, name);
            }

            var activatedFullyConnected = activationFunction.ApplyActivationFunction(fullyConnected, device);
            return Function.Alias(activatedFullyConnected, name);
        }

        private Function createResidualLayer2(Function net, int outputDimension, ActivationFunction activationFunction, DeviceDescriptor device, string name)
        {
            if (net.Output.Shape.Rank != 1)
            {
                int newDim = net.Output.Shape.Dimensions.Aggregate((d1, d2) => d1 * d2);
                net = CNTKLib.Reshape(net, new int[] { newDim });
            }
            //проброс входа мимо 1 слоя    
            var forwarding = net;
            if (outputDimension != net.Output.Shape[0])
            {
                var scales = new Parameter(new int[] { outputDimension, net.Output.Shape[0] }, _dataType, CNTKLib.UniformInitializer(CNTKLib.DefaultParamInitScale), _device);
                forwarding = CNTKLib.Times(scales, forwarding);
            }

            //создание 1 слоя
            net = createFullyConnectedLinearLayer(net, outputDimension, activationFunction, _device, "");
            //создание 2 слоя без функции активации
            net = createFullyConnectedLinearLayer(net, outputDimension, null, _device, "");
            //соединение с проброшенным входом
            net = CNTKLib.Plus(net, forwarding);

            if (activationFunction == null)
            {
                return Function.Alias(net, name);
            }
            net = activationFunction.ApplyActivationFunction(net, device);
            return Function.Alias(net, name);
        }


        #region LSTM
        /// <summary>
        /// Нормализация по документу https://www.microsoft.com/en-us/research/wp-content/uploads/2016/11/SelfLR.pdf
        /// </summary>
        /// <param name="input"></param>
        /// <param name="device"></param>
        /// <returns></returns>
        private Function selfStabilize<TElementType>(Function input, DeviceDescriptor device, string name)
        {
            bool isFloatType = typeof(TElementType) == typeof(float);
            Constant f, fInv;
            if (isFloatType)
            {
                f = Constant.Scalar(4.0f, device);
                fInv = Constant.Scalar(f.DataType, 1.0 / 4.0f);
            }
            else
            {
                f = Constant.Scalar(4.0, device);
                fInv = Constant.Scalar(f.DataType, 1.0 / 4.0);
            }

            var beta = CNTKLib.ElementTimes(
                fInv,
                CNTKLib.Log(
                    Constant.Scalar(f.DataType, 1.0) +
                    CNTKLib.Exp(CNTKLib.ElementTimes(f,
                        new Parameter(new NDShape(), f.DataType, 0.99537863 /* 1/f*ln(e^f-1)*/, device, "alpha")))),
                "beta");
            return Function.Alias(CNTKLib.ElementTimes(beta, input), name);
        }

        /// <summary>
        /// Создает ЛСТМ ячейку, которая реализует один шаг повторения в реккурентной сети.
        /// В качестве аргументов принимает предыдущие состояния ячейки(c - cell state) и выхода(h - hidden state)
        /// Возвращает кортеж нового состояния ячейки(c - cell state) и выхода(h - hidden state)        
        /// </summary>
        /// <typeparam name="TElementType">Тип данных. Поддерживается <seealso cref="float"/>, <seealso cref="double"/></typeparam>
        /// <param name="input">Вход в ЛСТМ (Х на шаге t)</param>
        /// <param name="prevOutput">Предыдущее состояние выхода ЛСТМ (h на шаге t-1)</param>
        /// <param name="prevCellState">Предыдущее состояние ячейки ЛСТМ (с на шаге t-1)</param>
        /// <param name="enableSelfStabilization">Указывает, будет ли применена самостабилизация к входам prevOutput и prevCellState</param>
        /// <param name="device">Устройтсво для расчетов</param>
        /// <returns>Функция (prev_h, prev_c, input) -> (h, c) которая реализует один шаг повторения ЛСТМ слоя</returns>
        private Tuple<Function, Function> LSTMCell<TElementType>(Variable input, Variable prevOutput,
            Variable prevCellState, bool enableSelfStabilization, bool enableShortcutConnections, DeviceDescriptor device)
        {
            int lstmInputDimension = input.Shape[0];
            int lstmOutputDimension = prevOutput.Shape[0];
            int lstmCellDimension = prevCellState.Shape[0];

            bool isFloatType = typeof(TElementType) == typeof(float);
            DataType dataType = isFloatType ? DataType.Float : DataType.Double;

            if (enableSelfStabilization)
            {
                prevOutput = selfStabilize<TElementType>(prevOutput, device, "");
                prevCellState = selfStabilize<TElementType>(prevCellState, device, "");
            }

            uint seed = 1;
            //создаем входную проекцию данных для ячейки из входа X и скрытого состояния H (из входных данных на шаге t)
            Func<int, int, Variable> createInput = (cellDim, outputDim) =>
            {
                var inputWeigths = new Parameter(new[] { cellDim, NDShape.InferredDimension }, dataType, CNTKLib.GlorotUniformInitializer(1.0, 1, 0, seed++), device);
                var inputBias = new Parameter(new[] { cellDim }, dataType, CNTKLib.GlorotUniformInitializer(1.0, 1, 0, seed++), device);
                var inputToCell = CNTKLib.Times(inputWeigths, input) + inputBias;

                var hiddenWeigths = new Parameter(new[] { cellDim, outputDim }, dataType, CNTKLib.GlorotUniformInitializer(1.0, 1, 0, seed++));
                var hiddenState = CNTKLib.Times(hiddenWeigths, prevOutput);

                var gateInput = CNTKLib.Plus(inputToCell, hiddenState);
                return gateInput;
            };

            Variable forgetProjection = createInput( lstmCellDimension, lstmOutputDimension);
            Variable inputProjection = createInput( lstmCellDimension, lstmOutputDimension);
            Variable candidateProjection = createInput( lstmCellDimension, lstmOutputDimension);
            Variable outputProjection = createInput( lstmCellDimension, lstmOutputDimension);

            Function forgetGate = CNTKLib.Sigmoid(forgetProjection); // вентиль "забывания" (из входных данных на шаге t)  
            Function inputGate = CNTKLib.Sigmoid(inputProjection); //вентиль входа (из входных данных на шаге t)         
            Function candidateGate = CNTKLib.Tanh(candidateProjection); //вентиль выбора кандидатов для запоминания в клеточном состоянии (из входных данных на шаге t)
            Function outputGate = CNTKLib.Sigmoid(outputProjection); //вентиль выхода (из входных данных на шаге t)  

            Function forgetState = CNTKLib.ElementTimes(prevCellState, forgetGate); //забываем то что нужно забыть в клеточном состоянии
            Function inputState = CNTKLib.ElementTimes(inputGate, candidateProjection); //получаем то что нужно сохранить в клеточном состоянии (из входных данных на шаге t) 
            Function cellState = CNTKLib.Plus(forgetState, inputState); //добавляем новую информацию в клеточное состояние

            Function h = CNTKLib.ElementTimes(outputGate, CNTKLib.Tanh(cellState)); //получаем выход/скрытое состояние
            Function c = cellState;
            if (lstmOutputDimension != lstmCellDimension)
            {
                Parameter P = new Parameter(new[] { lstmOutputDimension, lstmCellDimension }, dataType, CNTKLib.GlorotUniformInitializer(1.0, 1, 0, seed++));
                h = CNTKLib.Times(P, h);
            }
            if (enableShortcutConnections)
            {
                var forwarding = input;
                var inputDim = input.Shape[0];
                if (inputDim != lstmOutputDimension)
                {
                    var scales = new Parameter(new[] { lstmOutputDimension, inputDim }, dataType, CNTKLib.UniformInitializer(seed++));
                    forwarding = CNTKLib.Times(scales, input);
                }
                h = CNTKLib.Plus(h, forwarding);
            }

            return new Tuple<Function, Function>(h, c);
        }
        private Tuple<Function, Function> LSTMComponent<TElementType>(Variable input, NDShape outputShape,
            NDShape cellShape, Func<Variable, Function> recurrenceHookH, Func<Variable, Function> recurrenceHookC,
            bool enableSelfStabilization, bool enableShortcutConnections, DeviceDescriptor device)
        {
            var dh = Variable.PlaceholderVariable(outputShape, input.DynamicAxes);
            var dc = Variable.PlaceholderVariable(cellShape, input.DynamicAxes);

            var lstmCell = LSTMCell<TElementType>(input, dh, dc, enableSelfStabilization, enableShortcutConnections, device);
            var actualDh = recurrenceHookH(lstmCell.Item1);
            var actualDc = recurrenceHookC(lstmCell.Item2);

            (lstmCell.Item1).ReplacePlaceholders(new Dictionary<Variable, Variable> { { dh, actualDh }, { dc, actualDc } });
            return new Tuple<Function, Function>(lstmCell.Item1, lstmCell.Item2);
        }
        /// <summary>
        /// Добавляет слой LSTM. 
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="lstmDimension">Разрядность выходного слоя</param>        
        /// <param name="cellDimension">Разрядность внутреннего слоя ячейки памяти, если 0 - устанавливается разрдность выходного слоя</param>
        /// <param name="enableSelfStabilization">Если true, использовать самостабилизацию. По умолчанию включено.</param>
        /// <param name="isLastLstm">Указывает будет ли последний из слоев LSTM (следующие слои в сети нерекуррентные). Для того чтобы стыковать LSTM слои друг за другом, у всех слоев, кроме последнего, установить false</param>
        /// <param name="outputName">название слоя</param>
        /// <returns></returns>
        public Function AddLSTMLayer(int lstmDimension, int cellDimension = 0, bool enableSelfStabilization = true, bool enableShortcutConnections = true, bool isLastLstm = true, string outputName = "")
        {
            if (cellDimension == 0) cellDimension = lstmDimension;
            Func<Variable, Function> pastValueRecurrenceHook = (x) => CNTKLib.PastValue(x);

            var net = _net ?? _inputVariable;
            var lstm = LSTMComponent<T>(net, new int[] { lstmDimension }, new int[] { cellDimension },
                    pastValueRecurrenceHook, pastValueRecurrenceHook, enableSelfStabilization, enableShortcutConnections, _device)
                .Item1;

            _net = isLastLstm ? CNTKLib.SequenceLast(lstm) : lstm;
            ConfigurationDescription += $"-LSTM(C={cellDimension}H={lstmDimension})";
            return Function.Alias(_net, outputName);
        }

        #endregion

        #region convolution
        private Function createConvolutionLayer2D(Variable input, int kernelWidth, int kernelHeight, int outFeatureMapCount, int hStride, int vStride, Padding padding, ActivationFunction activationFunction, DeviceDescriptor device, string name)
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

            var convMap = new Parameter(new int[] { kernelWidth, kernelHeight, 1, outFeatureMapCount }, _dataType, CNTKLib.GlorotUniformInitializer(), device);
            var convolution = CNTKLib.Convolution(convMap, input, new int[] { hStride, vStride, 1 }, new bool[] { true }, paddingVector);
            if (activationFunction == null)
            {
                return Function.Alias(convolution, name);
            }

            var activatedConvolution = activationFunction.ApplyActivationFunction(convolution, device);
            return Function.Alias(activatedConvolution, name);
        }

        private Function createPoolingLayer2D(Variable input, int poolingWindowWidth, int poolingWindowHeight, int hStride, int vStride, PoolingType poolingType, string name)
        {
            var pooling = CNTKLib.Pooling(input, poolingType, new int[] { poolingWindowWidth, poolingWindowHeight }, new int[] { hStride, vStride }, new bool[] { true });
            return CNTKLib.Alias(pooling, name);
        }
        /// <summary>
        /// Добавляет сверточный слой для двумерного вектора. Если предыдущий слой имеет не двумерный выход, выбрасывается исключение
        /// </summary>
        /// <param name="kernelWidth">Ширина ядра свертки (столбцы в двумерной матрице)</param>
        /// <param name="kernelHeight">Высота ядра свертки (строки в двумерной матрице)</param>
        /// <param name="outFeatureMapCount">Разрядность выходной ячейки после свертки</param>
        /// <param name="activationFunction">Функция активации для выходного слоя. Если не требуется - передать null</param>
        /// <param name="hStride">Шаг спещения окна свертки по горизонтали (по столбцам матрицы)</param>
        /// <param name="vStride">Шаг смещения окна свертки по вертикали (по строкам матрицы)</param>
        /// <param name="name"></param>
        public void AddConvolution2D(int kernelWidth, int kernelHeight, int outFeatureMapCount = 1, ActivationFunction activationFunction = null, int hStride = 1, int vStride = 1, Padding padding = Padding.Valid, string name = "")
        {
            var net = _net ?? InputVariable;
            _net = createConvolutionLayer2D(net, kernelWidth, kernelHeight, outFeatureMapCount, hStride, vStride, padding, activationFunction, _device, name);
            ConfigurationDescription += $"-Conv2D({kernelWidth}x{kernelHeight})";
        }
        /// <summary>
        /// Добавляет пуллинг слой для двумерного вектора. Если предыдущий слой имеет не двумерный выход, выбрасывается исключение
        /// </summary>
        /// <param name="poolingWindowWidth">Ширина окна пуллинга</param>
        /// <param name="poolingWindowHeight">Высота окна пуллинга</param>
        /// <param name="hStride">Шаг смещения окна пуллинга по горизонтали (по столбцам матрицы)</param>
        /// <param name="vStride">Шаг смещения окна пуллинга по вертикали (по строкам матрицы)</param>
        /// <param name="poolingType">Тип пуллинга. Максимальный или средний</param>
        /// <param name="name"></param>
        public void AddPooling2D(int poolingWindowWidth, int poolingWindowHeight, int hStride = 1, int vStride = 1, PoolingType poolingType = PoolingType.Max, string name = "")
        {
            var net = _net ?? InputVariable;
            _net = createPoolingLayer2D(net, poolingWindowWidth, poolingWindowHeight, hStride, vStride, poolingType, name);
            ConfigurationDescription += $"-{poolingType.ToString()}Pooling({poolingWindowWidth}x{poolingWindowHeight})";
        }
        #endregion

        /// <summary>
        /// Описание конфигурации сети
        /// </summary>
        public string ConfigurationDescription { get; private set; }


        /// <summary>
        /// Инициализирeует нейросеть с размерностью входного вектора без слоев
        /// </summary>
        /// <param name="inputDimension">Разрядность входного вектора сети (входных данных)</param>
        /// <param name="random">Генератор случайных чисел (для первичной инициализации весов)</param>
        /// <param name="device">Устройство на котором создается сеть</param>   
        [Obsolete("В дальнейшем удалится входной параметр random")]
        public Model(int inputDimension, Random random, DeviceDescriptor device)
        {
            _inputVariable = Variable.InputVariable(new int[] { inputDimension }, _dataType, "input");
            _device = device;
            ConfigurationDescription = $"INPUT{inputDimension}";
        }
        /// <summary>
        /// Инициализирeует нейросеть с размерностью входного вектора без слоев
        /// </summary>
        /// <param name="inputDimension">Разрядность входного вектора сети (входных данных)</param>
        /// <param name="device">Устройство на котором создается сеть</param>
        public Model(DeviceDescriptor device, int[] inputShape)
        {
            _dataType = typeof(T) == typeof(float) ? DataType.Float : DataType.Double;
            _inputVariable = Variable.InputVariable(inputShape, _dataType, "input");
            _device = device;
            var shape = "";
            inputShape.ToList().ForEach(p =>
            {
                shape += p.ToString() + "x";
            });
            shape = shape.Substring(0, shape.Length - 1);
            ConfigurationDescription = $"INPUT{shape}";
        }
        /// <summary>
        /// Добавляет полносвязный слой в сеть с заданной функцией активации
        /// </summary>
        /// <param name="dimension">Количество нейронов в слое (разрядность слоя)</param>
        /// <param name="activationFunction">Функция активации. Если не требуется - передать null</param>
        public void AddFullyConnectedLayer(int dimension, ActivationFunction activationFunction, string name = "")
        {
            var net = _net ?? _inputVariable;
            _net = createFullyConnectedLinearLayer(net, dimension, activationFunction, _device, name);
            ConfigurationDescription += $"-{dimension}";
        }
        /// <summary>
        /// Применяет функцию дропаут к последнему добавленному слою
        /// </summary>
        /// <param name="dropoutRate">Доля отключаемых нейронов в слое</param>
        public void AddDropout(double dropoutRate)
        {
            if (_net != null)
            {
                _net = CNTKLib.Dropout(_net, dropoutRate);
                ConfigurationDescription += $"-DO({dropoutRate})";
            }
        }
        /// <summary>
        /// Добавляет слой батч-нормализации
        /// </summary>
        public void AddBatchNormalization()
        {
            var net = _net ?? _inputVariable;
            var scale = new Parameter(net.Output.Shape, net.Output.DataType, 1, _device);
            var bias = new Parameter(net.Output.Shape, net.Output.DataType, 0, _device);
            var runningMean = new Constant(net.Output.Shape, net.Output.DataType, 0, _device);
            var runningInvStd = new Constant(net.Output.Shape, net.Output.DataType, 0, _device);
            var runningCount = new Constant(new int[] { 1 }, net.Output.DataType, 0, _device);
            _net = CNTKLib.BatchNormalization(net.Output, scale, bias, runningMean, runningInvStd, runningCount, false);
            ConfigurationDescription += $"-BN";
        }
        public void AddResidualLayer2(int dimension, ActivationFunction activationFunction, string name = "")
        {
            var net = _net ?? _inputVariable;
            _net = createResidualLayer2(net, dimension, activationFunction, _device, name);
            ConfigurationDescription += $"-Res2({dimension})";

        }
        /// <summary>
        /// Сконфигурированная сеть
        /// </summary>
        /// <returns></returns>
        public Function NeuralNetCntk
        {
            get { return _net; }
        }
        /// <summary>
        /// Входная переменная сети
        /// </summary>
        public Variable InputVariable
        {
            get
            {
                return _inputVariable;
            }
        }
        /// <summary>
        /// Возвращает выходную переменную сети
        /// </summary>
        public Variable OutputVariable(bool isReccurent)
        {
            return isReccurent ? Variable.InputVariable(_net.Output.Shape, _net.Output.DataType, "output", new List<Axis> { Axis.DefaultBatchAxis() })
                : Variable.InputVariable(_net.Output.Shape, _net.Output.DataType, "output");
        }


    }
    // <summary>
    /// Полносвязная нейросеть прямого распространения
    /// </summary>
    [Obsolete]
    public class ModelRC
    {
        private Function _net;
        private Variable _inputVariable;
        private DeviceDescriptor _device;

        /// <summary>
        /// Создает полносвязный слой с заданной функцией активации
        /// </summary>
        /// <param name="input">Входная переменная(слой) заданной разрядности</param>
        /// <param name="outputDim">Выходная разрядность(кол-во нейронов)</param>
        /// <param name="activationFunction">Функция активации</param>
        /// <param name="device">Устройство на котором производится расчет</param>
        /// <param name="name">Имя слоя</param>
        /// <returns></returns>
        private Function createFullyConnectedLinearLayer(Variable input, int outputDim, ActivationFunction activationFunction, DeviceDescriptor device, string name)
        {
            if (input.Shape.Rank != 1)
            {
                // если данные не одномерные разворачиваем входной тензор в вектор
                int newDim = input.Shape.Dimensions.Aggregate((d1, d2) => d1 * d2);
                input = CNTKLib.Reshape(input, new int[] { newDim });
            }

            var inputDim = input.Shape[0];
            var weight = new Parameter(new int[] { outputDim, inputDim }, DataType.Double, CNTKLib.GlorotUniformInitializer(
                CNTKLib.DefaultParamInitScale,
                    CNTKLib.SentinelValueForInferParamInitRank,
                    CNTKLib.SentinelValueForInferParamInitRank, 1), device);
            var bias = new Parameter(new int[] { outputDim }, DataType.Double, 0, device);
            var fullyConnected = CNTKLib.Times(weight, input) + bias;

            if (activationFunction == null)
            {
                return Function.Alias(fullyConnected, name);
            }

            var activatedFullyConnected = activationFunction.ApplyActivationFunction(fullyConnected, device);
            return Function.Alias(activatedFullyConnected, name);
        }

        private Function createResidualLayer2(Function net, int outputDimension, ActivationFunction activationFunction, DeviceDescriptor device, string name)
        {
            if (net.Output.Shape.Rank != 1)
            {
                int newDim = net.Output.Shape.Dimensions.Aggregate((d1, d2) => d1 * d2);
                net = CNTKLib.Reshape(net, new int[] { newDim });
            }
            //проброс входа мимо 1 слоя    
            var forwarding = net;
            if (outputDimension != net.Output.Shape[0])
            {
                var scales = new Parameter(new int[] { outputDimension, net.Output.Shape[0] }, net.Output.DataType, CNTKLib.UniformInitializer(CNTKLib.DefaultParamInitScale), _device);
                forwarding = CNTKLib.Times(scales, forwarding);
            }            

            //создание 1 слоя
            net = createFullyConnectedLinearLayer(net, outputDimension, activationFunction, _device, "");
            //создание 2 слоя без функции активации
            net = createFullyConnectedLinearLayer(net, outputDimension, null, _device, "");
            //соединение с проброшенным входом
            net = CNTKLib.Plus(net, forwarding);

            if (activationFunction == null)
            {
                return Function.Alias(net, name);
            }
            net = activationFunction.ApplyActivationFunction(net, device);
            return Function.Alias(net, name);
        }


        #region LSTM
        /// <summary>
        /// Нормализация по документу https://www.microsoft.com/en-us/research/wp-content/uploads/2016/11/SelfLR.pdf
        /// </summary>
        /// <param name="input"></param>
        /// <param name="device"></param>
        /// <returns></returns>
        private Function selfStabilize<TElementType>(Function input, DeviceDescriptor device, string name)
        {
            bool isFloatType = typeof(TElementType) == typeof(float);
            Constant f, fInv;
            if (isFloatType)
            {
                f = Constant.Scalar(4.0f, device);
                fInv = Constant.Scalar(f.DataType, 1.0 / 4.0f);
            }
            else
            {
                f = Constant.Scalar(4.0, device);
                fInv = Constant.Scalar(f.DataType, 1.0 / 4.0);
            }

            var beta = CNTKLib.ElementTimes(
                fInv,
                CNTKLib.Log(
                    Constant.Scalar(f.DataType, 1.0) +
                    CNTKLib.Exp(CNTKLib.ElementTimes(f,
                        new Parameter(new NDShape(), f.DataType, 0.99537863 /* 1/f*ln(e^f-1)*/, device, "alpha")))),
                "beta");
            return Function.Alias(CNTKLib.ElementTimes(beta, input), name);
        }

        /// <summary>
        /// Создает ЛСТМ ячейку, которая реализует один шаг повторения в реккурентной сети.
        /// В качестве аргументов принимает предыдущие состояния ячейки(c - cell state) и выхода(h - hidden state)
        /// Возвращает кортеж нового состояния ячейки(c - cell state) и выхода(h - hidden state)        
        /// </summary>
        /// <typeparam name="TElementType">Тип данных. Поддерживается <seealso cref="float"/>, <seealso cref="double"/></typeparam>
        /// <param name="input">Вход в ЛСТМ (Х на шаге t)</param>
        /// <param name="prevOutput">Предыдущее состояние выхода ЛСТМ (h на шаге t-1)</param>
        /// <param name="prevCellState">Предыдущее состояние ячейки ЛСТМ (с на шаге t-1)</param>
        /// <param name="enableSelfStabilization">Указывает, будет ли применена самостабилизация к входам prevOutput и prevCellState</param>
        /// <param name="device">Устройтсво для расчетов</param>
        /// <returns>Функция (prev_h, prev_c, input) -> (h, c) которая реализует один шаг повторения ЛСТМ слоя</returns>
        private Tuple<Function, Function> LSTMCell<TElementType>(Variable input, Variable prevOutput,
            Variable prevCellState, bool enableSelfStabilization, bool enableShortcutConnections, DeviceDescriptor device)
        {
            int lstmOutputDimension = prevOutput.Shape[0];
            int lstmCellDimension = prevCellState.Shape[0];
            bool hasDifferentOutputAndCellDimension = lstmCellDimension != lstmOutputDimension;

            bool isFloatType = typeof(TElementType) == typeof(float);
            DataType dataType = isFloatType ? DataType.Float : DataType.Double;

            if (enableSelfStabilization)
            {
                prevOutput = selfStabilize<TElementType>(prevOutput, device, "");
                prevCellState = selfStabilize<TElementType>(prevCellState, device, "");
            }

            uint seed = CNTKLib.GenerateRandomSeed();
            //создаем входную проекцию данных из входа X и скрытого состояния H (из входных данных на шаге t)
            Func<int, Variable> createInput = (outputDim) =>
            {
                var inputWeigths = new Parameter(new[] { outputDim, NDShape.InferredDimension }, dataType, CNTKLib.GlorotUniformInitializer(1.0, 1, 0, seed++), device);
                var inputBias = new Parameter(new[] { outputDim }, dataType, CNTKLib.GlorotUniformInitializer(1.0, 1, 0, seed++), device);
                var inputToCell = CNTKLib.Times(inputWeigths, input) + inputBias;

                var gateInput = CNTKLib.Plus(inputToCell, prevOutput);
                return gateInput;
            };

            Func<int, Variable, Variable> createProjection = (targetDim, variableNeedsToProjection) =>
            {
                var cellWeigths = new Parameter(new[] { targetDim, NDShape.InferredDimension }, dataType, CNTKLib.GlorotUniformInitializer(1.0, 1, 0, seed++), device);
                var projection = CNTKLib.Times(cellWeigths, variableNeedsToProjection);
                return projection;
            };

            Variable forgetProjection = createInput(lstmOutputDimension);
            Variable inputProjection = createInput(lstmOutputDimension);
            Variable candidateProjection = createInput(lstmOutputDimension);
            Variable outputProjection = createInput(lstmOutputDimension);

            Function forgetGate = CNTKLib.Sigmoid(forgetProjection); // вентиль "забывания" (из входных данных на шаге t)  
            Function inputGate = CNTKLib.Sigmoid(inputProjection); //вентиль входа (из входных данных на шаге t)         
            Function candidateGate = CNTKLib.Tanh(candidateProjection); //вентиль выбора кандидатов для запоминания в клеточном состоянии (из входных данных на шаге t)
            Function outputGate = CNTKLib.Sigmoid(outputProjection); //вентиль выхода (из входных данных на шаге t)  

            forgetGate = hasDifferentOutputAndCellDimension ? createProjection(lstmCellDimension, forgetGate) : (Variable)forgetGate;
            Function forgetState = CNTKLib.ElementTimes(prevCellState, forgetGate); //забываем то что нужно забыть в клеточном состоянии

            Function inputState = CNTKLib.ElementTimes(inputGate, candidateProjection); //получаем то что нужно сохранить в клеточном состоянии (из входных данных на шаге t) 
            inputState = hasDifferentOutputAndCellDimension ? createProjection(lstmCellDimension, inputState) : (Variable)inputState;
            Function cellState = CNTKLib.Plus(forgetState, inputState); //добавляем новую информацию в клеточное состояние

            Variable cellToOutputProjection = hasDifferentOutputAndCellDimension ? createProjection(lstmOutputDimension, cellState) : (Variable)cellState;
            Function h = CNTKLib.ElementTimes(outputGate, CNTKLib.Tanh(cellToOutputProjection)); //получаем выход/скрытое состояние
            Function c = cellState;
            
            if (enableShortcutConnections)
            {
                var forwarding = input;
                var inputDim = input.Shape[0];
                if (inputDim != lstmOutputDimension)
                {
                    var scales = new Parameter(new[] { lstmOutputDimension, inputDim }, dataType, CNTKLib.UniformInitializer(seed++));
                    forwarding = CNTKLib.Times(scales, input);
                }
                h = CNTKLib.Plus(h, CNTKLib.Tanh(forwarding));
            }

            return new Tuple<Function, Function>(h, c);
        }
        private Tuple<Function, Function> LSTMComponent<TElementType>(Variable input, NDShape outputShape,
            NDShape cellShape, Func<Variable, Function> recurrenceHookH, Func<Variable, Function> recurrenceHookC,
            bool enableSelfStabilization, bool enableShortcutConnections, DeviceDescriptor device)
        {
            var dh = Variable.PlaceholderVariable(outputShape, input.DynamicAxes);
            var dc = Variable.PlaceholderVariable(cellShape, input.DynamicAxes);            

            var lstmCell = LSTMCell<TElementType>(input, dh, dc, enableSelfStabilization, enableShortcutConnections, device);
            var actualDh = recurrenceHookH(lstmCell.Item1);
            var actualDc = recurrenceHookC(lstmCell.Item2);

            (lstmCell.Item1).ReplacePlaceholders(new Dictionary<Variable, Variable> { { dh, actualDh }, { dc, actualDc } });
            return new Tuple<Function, Function>(lstmCell.Item1, lstmCell.Item2);
        }
        /// <summary>
        /// Добавляет слой LSTM. 
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="lstmDimension">Разрядность выходного слоя</param>        
        /// <param name="cellDimension">Разрядность внутреннего слоя ячейки памяти, если 0 - устанавливается разрдность выходного слоя</param>
        /// <param name="enableSelfStabilization">Если true, использовать самостабилизацию. По умолчанию включено.</param>
        /// <param name="isLastLstm">Указывает будет ли последний из слоев LSTM (следующие слои в сети нерекуррентные). Для того чтобы стыковать LSTM слои друг за другом, у всех слоев, кроме последнего, установить false</param>
        /// <param name="outputName">название слоя</param>
        /// <returns></returns>
        public Function AddLSTMLayer<T>(int lstmDimension, int cellDimension = 0, bool enableSelfStabilization = true, bool enableShortcutConnections = true, bool isLastLstm = true, string outputName = "")
        {
            if (cellDimension == 0) cellDimension = lstmDimension;
            Func<Variable, Function> pastValueRecurrenceHook = (x) => CNTKLib.PastValue(x);

            var net = _net ?? _inputVariable;
            var lstm = LSTMComponent<T>(net, new int[] { lstmDimension }, new int[] { cellDimension },
                    pastValueRecurrenceHook, pastValueRecurrenceHook, enableSelfStabilization, enableShortcutConnections, _device)
                .Item1;

            _net = isLastLstm ? CNTKLib.SequenceLast(lstm) : lstm;
            ConfigurationDescription += $"-LSTM(C={cellDimension}H={lstmDimension})";
            return Function.Alias(_net, outputName);
        }

        #endregion

        #region convolution
        private Function createConvolutionLayer2D(Variable input, int kernelWidth, int kernelHeight, int outFeatureMapCount, int hStride, int vStride, Padding padding, ActivationFunction activationFunction, DeviceDescriptor device, string name)
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

            var convMap = new Parameter(new int[] { kernelWidth, kernelHeight, 1, outFeatureMapCount }, DataType.Double, CNTKLib.GlorotUniformInitializer(), device);
            var convolution = CNTKLib.Convolution(convMap, input, new int[] { hStride, vStride, 1 }, new bool[] { true }, paddingVector);
            if (activationFunction == null)
            {
                return Function.Alias(convolution, name);
            }

            var activatedConvolution = activationFunction.ApplyActivationFunction(convolution, device);
            return Function.Alias(activatedConvolution, name);
        }

        private Function createPoolingLayer2D(Variable input, int poolingWindowWidth, int poolingWindowHeight, int hStride, int vStride, PoolingType poolingType, string name)
        {
            var pooling = CNTKLib.Pooling(input, poolingType, new int[] { poolingWindowWidth, poolingWindowHeight }, new int[] { hStride, vStride }, new bool[] { true });
            return CNTKLib.Alias(pooling, name);
        }
        /// <summary>
        /// Добавляет сверточный слой для двумерного вектора. Если предыдущий слой имеет не двумерный выход, выбрасывается исключение
        /// </summary>
        /// <param name="kernelWidth">Ширина ядра свертки (столбцы в двумерной матрице)</param>
        /// <param name="kernelHeight">Высота ядра свертки (строки в двумерной матрице)</param>
        /// <param name="outFeatureMapCount">Разрядность выходной ячейки после свертки</param>
        /// <param name="activationFunction">Функция активации для выходного слоя. Если не требуется - передать null</param>
        /// <param name="hStride">Шаг спещения окна свертки по горизонтали (по столбцам матрицы)</param>
        /// <param name="vStride">Шаг смещения окна свертки по вертикали (по строкам матрицы)</param>
        /// <param name="name"></param>
        public void AddConvolution2D(int kernelWidth, int kernelHeight, int outFeatureMapCount = 1, ActivationFunction activationFunction = null, int hStride = 1, int vStride = 1, Padding padding = Padding.Valid, string name = "")
        {
            var net = _net ?? InputVariable;
            _net = createConvolutionLayer2D(net, kernelWidth, kernelHeight, outFeatureMapCount, hStride, vStride, padding, activationFunction, _device, name);
            ConfigurationDescription += $"-Conv2D({kernelWidth}x{kernelHeight})";
        }
        /// <summary>
        /// Добавляет пуллинг слой для двумерного вектора. Если предыдущий слой имеет не двумерный выход, выбрасывается исключение
        /// </summary>
        /// <param name="poolingWindowWidth">Ширина окна пуллинга</param>
        /// <param name="poolingWindowHeight">Высота окна пуллинга</param>
        /// <param name="hStride">Шаг смещения окна пуллинга по горизонтали (по столбцам матрицы)</param>
        /// <param name="vStride">Шаг смещения окна пуллинга по вертикали (по строкам матрицы)</param>
        /// <param name="poolingType">Тип пуллинга. Максимальный или средний</param>
        /// <param name="name"></param>
        public void AddPooling2D(int poolingWindowWidth, int poolingWindowHeight, int hStride = 1, int vStride = 1, PoolingType poolingType = PoolingType.Max, string name = "")
        {
            var net = _net ?? InputVariable;
            _net = createPoolingLayer2D(net, poolingWindowWidth, poolingWindowHeight, hStride, vStride, poolingType, name);
            ConfigurationDescription += $"-{poolingType.ToString()}Pooling({poolingWindowWidth}x{poolingWindowHeight})";
        }
        #endregion

        /// <summary>
        /// Описание конфигурации сети
        /// </summary>
        public string ConfigurationDescription { get; private set; }


        /// <summary>
        /// Инициализирeует нейросеть с размерностью входного вектора без слоев
        /// </summary>
        /// <param name="inputDimension">Разрядность входного вектора сети (входных данных)</param>
        /// <param name="random">Генератор случайных чисел (для первичной инициализации весов)</param>
        /// <param name="device">Устройство на котором создается сеть</param>   
        [Obsolete("В дальнейшем удалится входной параметр random")]
        public ModelRC(int inputDimension, Random random, DeviceDescriptor device)
        {
            _inputVariable = Variable.InputVariable(new int[] { inputDimension }, DataType.Double, "input");
            _device = device;
            ConfigurationDescription = $"INPUT{inputDimension}";
        }
        /// <summary>
        /// Инициализирeует нейросеть с размерностью входного вектора без слоев
        /// </summary>
        /// <param name="inputDimension">Разрядность входного вектора сети (входных данных)</param>
        /// <param name="device">Устройство на котором создается сеть</param>
        public ModelRC(DeviceDescriptor device, int[] inputShape)
        {
            var sequencesAxis = new Axis();
            _inputVariable = Variable.InputVariable(inputShape, DataType.Double, "input");
            _device = device;
            var shape = "";
            inputShape.ToList().ForEach(p =>
            {
                shape += p.ToString() + "x";
            });
            shape = shape.Substring(0, shape.Length - 1);
            ConfigurationDescription = $"INPUT{shape}";
        }
        /// <summary>
        /// Добавляет полносвязный слой в сеть с заданной функцией активации
        /// </summary>
        /// <param name="dimension">Количество нейронов в слое (разрядность слоя)</param>
        /// <param name="activationFunction">Функция активации. Если не требуется - передать null</param>
        public void AddFullyConnectedLayer(int dimension, ActivationFunction activationFunction, string name = "")
        {
            var net = _net ?? _inputVariable;
            _net = createFullyConnectedLinearLayer(net, dimension, activationFunction, _device, name);
            ConfigurationDescription += $"-{dimension}";
        }
        /// <summary>
        /// Применяет функцию дропаут к последнему добавленному слою
        /// </summary>
        /// <param name="dropoutRate">Доля отключаемых нейронов в слое</param>
        public void AddDropout(double dropoutRate)
        {
            if (_net != null)
            {
                _net = CNTKLib.Dropout(_net, dropoutRate);
                ConfigurationDescription += $"-DO({dropoutRate})";
            }
        }
        /// <summary>
        /// Добавляет слой батч-нормализации
        /// </summary>
        public void AddBatchNormalization()
        {
            var net = _net ?? _inputVariable;
            var scale = new Parameter(net.Output.Shape, net.Output.DataType, 1, _device);
            var bias = new Parameter(net.Output.Shape, net.Output.DataType, 0, _device);
            var runningMean = new Constant(net.Output.Shape, net.Output.DataType, 0, _device);
            var runningInvStd = new Constant(net.Output.Shape, net.Output.DataType, 0, _device);
            var runningCount = new Constant(new int[] { 1 }, net.Output.DataType, 0, _device);
            _net = CNTKLib.BatchNormalization(net.Output, scale, bias, runningMean, runningInvStd, runningCount, false);
            ConfigurationDescription += $"-BN";
        }
        public void AddResidualLayer2(int dimension, ActivationFunction activationFunction, string name = "")
        {
            var net = _net ?? _inputVariable;
            _net = createResidualLayer2(net, dimension, activationFunction, _device, name);
            ConfigurationDescription += $"-Res2({dimension})";

        }
        /// <summary>
        /// Сконфигурированная сеть
        /// </summary>
        /// <returns></returns>
        public Function NeuralNetCntk
        {
            get { return _net; }
        }
        /// <summary>
        /// Входная переменная сети
        /// </summary>
        public Variable InputVariable
        {
            get
            {
                return _inputVariable;
            }
        }
        /// <summary>
        /// Возвращает выходную переменную сети
        /// </summary>
        public Variable OutputVariable(bool isReccurent)
        {
            return isReccurent ? Variable.InputVariable(_net.Output.Shape, _net.Output.DataType, "output", new List<Axis> { Axis.DefaultBatchAxis() })
                : Variable.InputVariable(_net.Output.Shape, _net.Output.DataType, "output");
        }


    }
}


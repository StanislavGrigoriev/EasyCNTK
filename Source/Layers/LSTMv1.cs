using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using CNTK;

namespace EasyCNTK.Layers
{
    /// <summary>
    /// Реализует слой LSTM.
    /// Клеточное состояние (С) имеет независимую размерность, все гейты имеют размерность выхода (H), масштабирование производится непосредственно при записи в клеточное состояние.
    /// Вход (X[t]) масштабируется к выходу (H[t-1]) и суммируется (X[t]+H[t-1]), ячейка памяти (С) масштабируется к выходу (H). 
    /// </summary>
    public sealed class LSTMv1 : Layer
    {
        private int _lstmOutputDim;
        private int _lstmCellDim;        
        private bool _useShortcutConnections;
        private bool _isLastLstmLayer;
        private string _name;
        private Layer _selfStabilizerLayer;

        /// <summary>
        /// Создает ЛСТМ ячейку, которая реализует один шаг повторения в реккурентной сети.
        /// В качестве аргументов принимает предыдущие состояния ячейки(c - cell state) и выхода(h - hidden state).
        /// Возвращает кортеж нового состояния ячейки(c - cell state) и выхода(h - hidden state).      
        /// </summary>
        /// <param name="input">Вход в ЛСТМ (Х на шаге t)</param>
        /// <param name="prevOutput">Предыдущее состояние выхода ЛСТМ (h на шаге t-1)</param>
        /// <param name="prevCellState">Предыдущее состояние ячейки ЛСТМ (с на шаге t-1)</param>
        /// <param name="useSelfStabilization">Указывает, будет ли применена самостабилизация к входам prevOutput и prevCellState</param>
        /// <param name="selfStabilizerLayer">Слой, реализующий самостабилизацию</param>
        /// <param name="device">Устройтсво для расчетов</param>
        /// <returns></returns>
        private static Tuple<Function, Function> LSTMCell(Variable input, Variable prevOutput,
            Variable prevCellState,  bool useShortcutConnections, Layer selfStabilizerLayer, DeviceDescriptor device)
        {
            int lstmOutputDimension = prevOutput.Shape[0];
            int lstmCellDimension = prevCellState.Shape[0];
            bool hasDifferentOutputAndCellDimension = lstmCellDimension != lstmOutputDimension;

            DataType dataType = input.DataType;

            if (selfStabilizerLayer != null)
            {
                prevOutput = selfStabilizerLayer.Create(prevOutput, device);
                prevCellState = selfStabilizerLayer.Create(prevCellState, device);
            }

            uint seed = CNTKLib.GetRandomSeed();
            //создаем входную проекцию данных из входа X[t] и скрытого состояния H[t-1]
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

            if (useShortcutConnections)
            {
                var forwarding = input;
                var inputDim = input.Shape[0];
                if (inputDim != lstmOutputDimension)
                {
                    var scales = new Parameter(new[] { lstmOutputDimension, inputDim }, dataType, CNTKLib.UniformInitializer(seed++), device);
                    forwarding = CNTKLib.Times(scales, input);
                }
                h = CNTKLib.Plus(h, forwarding);
            }

            return new Tuple<Function, Function>(h, c);
        }
        private static Tuple<Function, Function> LSTMComponent(Variable input, NDShape outputShape,
            NDShape cellShape, Func<Variable, Function> recurrenceHookH, Func<Variable, Function> recurrenceHookC,
            bool useShortcutConnections, Layer selfStabilizerLayer, DeviceDescriptor device)
        {
            var dh = Variable.PlaceholderVariable(outputShape, input.DynamicAxes);
            var dc = Variable.PlaceholderVariable(cellShape, input.DynamicAxes);

            var lstmCell = LSTMCell(input, dh, dc, useShortcutConnections, selfStabilizerLayer, device);
            var actualDh = recurrenceHookH(lstmCell.Item1);
            var actualDc = recurrenceHookC(lstmCell.Item2);

            (lstmCell.Item1).ReplacePlaceholders(new Dictionary<Variable, Variable> { { dh, actualDh }, { dc, actualDc } });
            return new Tuple<Function, Function>(lstmCell.Item1, lstmCell.Item2);
        }
        /// <summary>
        /// Создает слой LSTM. 
        /// Клеточное состояние (С) имеет независимую размерность, все гейты имеют размерность выхода (H), масштабирование производится непосредвенно при записи в клеточное состояние.
        /// Вход (X[t]) масштабируется к выходу (H[t-1]) и суммируется (X[t]+H[t-1]), ячейка памяти (С) масштабируется к выходу (H). 
        /// </summary>
        /// <param name="input">Вход (X)</param>
        /// <param name="lstmDimension">Разрядность выходного слоя (H)</param>        
        /// <param name="cellDimension">Разрядность внутреннего слоя ячейки памяти, если 0 - устанавливается разрядность выходного слоя (C)</param>
        /// <param name="useShortcutConnections">Если true, использовать проброс входа параллельно слою. По умолчанию включено.</param>
        /// <param name="selfStabilizerLayer">Слой, реализующий самостабилизацию</param>
        /// <param name="isLastLstm">Указывает, будет ли это последний из слоев LSTM (следующие слои в сети нерекуррентные). Для того чтобы стыковать LSTM слои друг за другом, у всех слоев, кроме последнего, нужно установить false</param>
        /// <param name="outputName">название слоя</param>
        /// <returns></returns>
        public static Function Build(Function input, int lstmDimension, DeviceDescriptor device, int cellDimension = 0, bool useShortcutConnections = true, bool isLastLstm = true, Layer selfStabilizerLayer = null, string outputName = "")
        {
            if (cellDimension == 0) cellDimension = lstmDimension;
            
            Func<Variable, Function> pastValueRecurrenceHook = (x) => CNTKLib.PastValue(x);

            var lstm = LSTMComponent(input, new int[] { lstmDimension }, new int[] { cellDimension },
                    pastValueRecurrenceHook, pastValueRecurrenceHook, useShortcutConnections, selfStabilizerLayer, device)
                .Item1;

            lstm = isLastLstm ? CNTKLib.SequenceLast(lstm) : lstm;
            return Function.Alias(lstm, outputName);
        }

        public override Function Create(Function input, DeviceDescriptor device)
        {
            return Build(input, _lstmOutputDim, device, _lstmCellDim, _useShortcutConnections, _isLastLstmLayer, _selfStabilizerLayer, _name);
        }
        /// <summary>
        /// Создает слой LSTM. 
        /// Клеточное состояние (С) имеет независимую размерность, все гейты имеют размерность выхода (H), масштабирование производится непосредвенно при записи в клеточное состояние.
        /// Вход (X[t]) масштабируется к выходу (H[t-1]), ячейка памяти (С) масштабируется к выходу (H). 
        /// </summary>
        /// <param name="lstmOutputDim">Разрядность выходного слоя (H)</param>        
        /// <param name="lstmCellDim">Разрядность внутреннего слоя ячейки памяти, если 0 - устанавливается разрядность выходного слоя (C)</param>
        /// <param name="useShortcutConnections">Если true, использовать проброс входа параллельно слою. По умолчанию включено.</param>
        /// <param name="selfStabilizerLayer">Слой, реализующий самостабилизацию</param>
        /// <param name="isLastLstm">Указывает, будет ли это последний из слоев LSTM (следующие слои в сети нерекуррентные). Для того чтобы стыковать LSTM слои друг за другом, у всех слоев, кроме последнего, нужно установить false</param>
        /// <param name="name"></param>
        public LSTMv1(int lstmOutputDim, int lstmCellDim = 0, bool useShortcutConnections = true, bool isLastlstm = true, Layer selfStabilizerLayer = null, string name = "LSTMv1")
        {
            _lstmOutputDim = lstmOutputDim;
            _lstmCellDim = lstmCellDim == 0 ? _lstmOutputDim : _lstmCellDim; ;
            _useShortcutConnections = useShortcutConnections;
            _isLastLstmLayer = isLastlstm;
            _selfStabilizerLayer = selfStabilizerLayer;
            _name = name;
        }

        public override string GetDescription()
        {
            return $"LSTMv1(C={_lstmCellDim}H={_lstmOutputDim}SC={_useShortcutConnections}SS={_selfStabilizerLayer?.GetDescription() ?? "none"})";
        }
    }
}

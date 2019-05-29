//
// Copyright (c) Stanislav Grigoriev. All rights reserved.
// grigorievstas9@gmail.com 
// https://github.com/StanislavGrigoriev/EasyCNTK
//
// Copyright (c) Microsoft. All rights reserved.
//
// Licensed under the MIT license. See LICENSE.txt file in the project root for full license information.
//
using System;
using System.Collections.Generic;
using CNTK;

namespace EasyCNTK.Layers
{
    /// <summary>
    /// Реализует слой LSTM.
    /// Клеточное состояние (С) имеет общую размерность - все гейты имеют размерность клеточного состояния, масштабирование производится только на входе и выходе  из ячейки.
    /// Вход (X[t]+H[t-1]) масштабируется к ячейке памяти (С[t]), ячейка памяти масштабируется к выходу (H[t])
    /// </summary>
    public sealed class LSTM : Layer
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
        /// <param name="useShortcutConnections">Указывает, следует ли создавать ShortcutConnections для этой ячейки</param>
        /// <param name="selfStabilizerLayer">Слой, реализующий самостабилизацию. Если не null -  будет применена самостабилизация к входам prevOutput и prevCellState </param>
        /// <param name="device">Устройтсво для расчетов</param>
        /// <returns>Функция (prev_h, prev_c, input) -> (h, c) которая реализует один шаг повторения ЛСТМ слоя</returns>
        private static Tuple<Function, Function> LSTMCell(Variable input, Variable prevOutput,
            Variable prevCellState, bool useShortcutConnections, Layer selfStabilizerLayer, DeviceDescriptor device)
        {
            int lstmOutputDimension = prevOutput.Shape[0];
            int lstmCellDimension = prevCellState.Shape[0];

            DataType dataType = input.DataType;

            if (selfStabilizerLayer != null)
            {
                prevOutput = selfStabilizerLayer.Create(prevOutput, device);
                prevCellState = selfStabilizerLayer.Create(prevCellState, device);
            }

            uint seed = CNTKLib.GetRandomSeed();
            //создаем входную проекцию данных для ячейки из входа X[t] и скрытого состояния H[t-1]
            Func<int, int, Variable> createInput = (cellDim, hiddenDim) =>
            {
                var inputWeigths = new Parameter(new[] { cellDim, NDShape.InferredDimension }, dataType, CNTKLib.GlorotUniformInitializer(1.0, 1, 0, seed++), device);
                var inputBias = new Parameter(new[] { cellDim }, dataType, CNTKLib.GlorotUniformInitializer(1.0, 1, 0, seed++), device);
                var inputToCell = CNTKLib.Times(inputWeigths, input) + inputBias;

                var hiddenWeigths = new Parameter(new[] { cellDim, hiddenDim }, dataType, CNTKLib.GlorotUniformInitializer(1.0, 1, 0, seed++), device);
                var hiddenState = CNTKLib.Times(hiddenWeigths, prevOutput);

                var gateInput = CNTKLib.Plus(inputToCell, hiddenState);
                return gateInput;
            };

            Variable forgetProjection = createInput(lstmCellDimension, lstmOutputDimension);
            Variable inputProjection = createInput(lstmCellDimension, lstmOutputDimension);
            Variable candidateProjection = createInput(lstmCellDimension, lstmOutputDimension);
            Variable outputProjection = createInput(lstmCellDimension, lstmOutputDimension);

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
                Parameter scale = new Parameter(new[] { lstmOutputDimension, lstmCellDimension }, dataType, CNTKLib.GlorotUniformInitializer(1.0, 1, 0, seed++), device);
                h = CNTKLib.Times(scale, h);
            }
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
        /// Клеточное состояние (С) имеет общую размерность - все гейты имеют размерность клеточного состояния, масштабирование производится только на входе и выходе  из ячейки.
        /// Вход (X[t]+H[t-1]) масштабируется к ячейке памяти (С[t]), ячейка памяти масштабируется к выходу (H[t])
        /// </summary>
        /// <param name="input">Вход (X)</param>
        /// <param name="lstmDimension">Разрядность выходного слоя (H)</param>        
        /// <param name="cellDimension">Разрядность внутреннего слоя ячейки памяти, если 0 - устанавливается разрядность выходного слоя (C)</param>        
        /// <param name="useShortcutConnections">Если true, использовать проброс входа параллельно слою. По умолчанию включено.</param>
        /// <param name="selfStabilizerLayer">Слой, реализующий самостабилизацию. Если не null -  будет применена самостабилизация к входам prevOutput и prevCellState</param>
        /// <param name="isLastLstm">Указывает, будет ли это последний из слоев LSTM (следующие слои в сети нерекуррентные). Для того чтобы стыковать LSTM слои друг за другом, у всех слоев, кроме последнего, нужно установить false</param>
        /// <param name="outputName">название слоя</param>
        /// <returns></returns>
        public static Function Build(Function input, int lstmDimension, DeviceDescriptor device, int cellDimension = 0, bool useShortcutConnections = true, bool isLastLstm = true, Layer selfStabilizerLayer = null, string outputName = "")
        {
            if (cellDimension == 0) cellDimension = lstmDimension;            
            Func<Variable, Function> pastValueRecurrenceHook = (x) => CNTKLib.PastValue(x);

            var lstm = LSTMComponent(input, new int[] { lstmDimension }, new int[] { cellDimension },
                    pastValueRecurrenceHook, pastValueRecurrenceHook,  useShortcutConnections, selfStabilizerLayer, device)
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
        /// Клеточное состояние (С) имеет общую размерность - все гейты имеют размерность клеточного состояния, масштабирование производится только на входе и выходе  из ячейки.
        /// Вход (X[t]+H[t-1]) масштабируется к ячейке памяти (С[t]), ячейка памяти масштабируется к выходу (H[t])
        /// </summary>
        /// <param name="lstmOutputDim">Разрядность выходного слоя (H)</param>        
        /// <param name="lstmCellDim">Разрядность внутреннего слоя ячейки памяти, если 0 - устанавливается разрядность выходного слоя (C)</param>
        /// <param name="useShortcutConnections">Если true, использовать проброс входа параллельно слою. По умолчанию включено.</param>
        /// <param name="selfStabilizerLayer">Слой, реализующий самостабилизацию. Если не null -  будет применена самостабилизация к входам C[t-1] и H[t-1]</param>
        /// <param name="isLastLstm">Указывает, будет ли это последний из слоев LSTM (следующие слои в сети нерекуррентные). Для того чтобы стыковать LSTM слои друг за другом, у всех слоев, кроме последнего, нужно установить false</param>
        /// <param name="name"></param>
        public LSTM(int lstmOutputDim, int lstmCellDim = 0, bool useShortcutConnections = true, bool isLastLstm = true, Layer selfStabilizerLayer = null, string name = "LSTM")
        {
            _lstmOutputDim = lstmOutputDim;
            _lstmCellDim = lstmCellDim == 0 ? _lstmOutputDim : _lstmCellDim;            
            _useShortcutConnections = useShortcutConnections;
            _isLastLstmLayer = isLastLstm;
            _selfStabilizerLayer = selfStabilizerLayer;
            _name = name;
        }

        public override string GetDescription()
        {
            return $"LSTM(C={_lstmCellDim}H={_lstmOutputDim}SC={_useShortcutConnections}SS={_selfStabilizerLayer?.GetDescription() ?? "none"})";
        }
    }
}

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
using EasyCNTK.Learning.Optimizers;
using EasyCNTK.LossFunctions;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;

namespace EasyCNTK.Learning
{
    public static class LearnExtensions
    {
        #region Extensions for Function
        /// <summary>
        /// Обучает модель. Поддерживает реккурентные сети.
        /// </summary>
        /// <param name="source"></param>
        /// <param name="trainDataSelector">Селектор, позволяющий указать для каждой эпохи свой набор данных для обучения.</param>
        /// <param name="lossFunction">Функция потерь</param>
        /// <param name="evaluationFunction">Оценочная функция</param>
        /// <param name="optimizer">Оптимизатор, используемый для обучения</param>
        /// <param name="epochCount">Количество эпох обучения</param>
        /// <param name="isReccurentModel">Указывает, что требуется обучать реккурентную модель</param>
        /// <param name="device">Устройство для обучения</param>
        /// <param name="ruleUpdateLearningRate">Правило обновления скорости обучения. Входные параметры: эпоха, текущая скорость обучения. Выходные: новая скорость обучения.</param>
        /// <param name="actionPerEpoch">Произвольное действие, которое требуется выполнять каждую эпоху. Позволяет прервать процесс тренировки. Входные параметры: эпоха, loss-ошибка, evaluation-ошибка. 
        /// Выходные: true - прервать процесс тренировки, false - продолжить тренировку.
        /// Используется для осуществления логирования, отображения процесса обучения, сохранения промежуточных чекпоинтов модели и т.п.</param>
        /// <returns></returns>
        private static FitResult[] FitMultiOutput(this Function source,
           Func<int, IEnumerable<MinibatchMultiOutput>> trainDataSelector,
           Loss[] lossFunctions,
           Loss[] evaluationFunctions,
           Optimizer[] optimizers,
           int epochCount,
           bool isReccurentModel,
           DeviceDescriptor device,
           Func<int, double[], double[]> ruleUpdateLearningRate = null,
           Func<int, double[], double[], bool> actionPerEpoch = null)
        {
            int headCount = trainDataSelector(1).FirstOrDefault().Labels.Length;

            if (headCount != lossFunctions.Length) throw new ArgumentOutOfRangeException(nameof(lossFunctions), "Количество функций потерь не совпадает с количеством выходных голов модели.");
            if (headCount != evaluationFunctions.Length) throw new ArgumentOutOfRangeException(nameof(evaluationFunctions), "Количество оценочных функций не совпадает с количеством выходных голов модели.");
            if (headCount != optimizers.Length) throw new ArgumentOutOfRangeException(nameof(optimizers), "Количество оптимизаторов не совпадает с количеством выходных голов модели.");
            
            var inputVariable = source.Inputs.Single(p => p.Name.ToUpper() == "INPUT");

            var outputVariables = new Function[headCount];
            var losses = new Function[headCount];
            var evaluations = new Function[headCount];
            var learners = new Learner[headCount];
            var trainers = new Trainer[headCount];
            var learningRates = new double[headCount];

            for (int i = 0; i < headCount; i++)
            {
                var outputVariable = isReccurentModel 
                    ? Variable.InputVariable(source.Outputs[i].Shape, source.Outputs[i].DataType, "output", new List<Axis> { Axis.DefaultBatchAxis() })
                    : Variable.InputVariable(source.Outputs[i].Shape, source.Outputs[i].DataType, "output");
                var loss       = lossFunctions[i].GetLoss(source.Outputs[i], outputVariable, device);
                var evaluation = evaluationFunctions[i].GetLoss(source.Outputs[i], outputVariable, device);
                var learner    = optimizers[i].GetOptimizer(source.Outputs[i].ToFunction().Parameters());
                var trainer = CNTKLib.CreateTrainer(
                        source.Outputs[i],
                        loss,
                        evaluation,
                        new LearnerVector() { learner });

                outputVariables[i] = outputVariable;
                losses[i]          = loss;
                evaluations[i]     = evaluation;
                learners[i]        = learner;
                learningRates[i]   = optimizers[i].LearningRate;
                trainers[i]        = trainer;                
            }
            
            var factLosses = new List<double[]>(epochCount);
            var factEvals = new List<double[]>(epochCount);
            Stopwatch sw = new Stopwatch();
            sw.Start();
            for (int i = 1; i <= epochCount; i++)
            {
                var trainMinibatches = trainDataSelector(i);
                foreach (var miniBatch in trainMinibatches)
                {
                    for (int j = 0; j < headCount; j++)
                    {
                        trainers[j].TrainMinibatch(new Dictionary<Variable, Value>() { { inputVariable, miniBatch.Features }, { outputVariables[j], miniBatch.Labels[j] } }, false, device);
                    }
                }
                factLosses.Add(trainers.Select(p => p.PreviousMinibatchLossAverage()).ToArray());
                factEvals.Add(trainers.Select(p => p.PreviousMinibatchEvaluationAverage()).ToArray());

                bool needStopTraining = actionPerEpoch?.Invoke(i, factLosses[i - 1], factEvals [i - 1]) ?? false;
                if (needStopTraining)
                {
                    epochCount = i;
                    break;
                }

                if (ruleUpdateLearningRate != null)
                {
                    var proposaledLearningRate = ruleUpdateLearningRate(i, learningRates);
                    if (proposaledLearningRate.Length != headCount)
                        throw new ArgumentOutOfRangeException(nameof(ruleUpdateLearningRate), "Количество обновляемых скоростей обучения не соответсвует количеству выходных голов модели.");
                    for (int j = 0; j < headCount; j++)
                    {
                        if (proposaledLearningRate[j] != learningRates[j])
                        {
                            learners[j].SetLearningRateSchedule(new TrainingParameterScheduleDouble(learningRates[j]));
                            learningRates[j] = proposaledLearningRate[j];
                        }
                    }
                }
            }
            sw.Stop();

            return Enumerable.Range(0, headCount)
                .Select((indexHead, p) => new FitResult
                {
                    LossError = factLosses[factLosses.Count - 1][indexHead],
                    EvaluationError = factEvals[factEvals.Count - 1][indexHead],
                    Duration = sw.Elapsed,
                    EpochCount = epochCount,
                    LossCurve = factLosses.Select(q => q[indexHead]).ToList(),
                    EvaluationCurve = factEvals.Select(q => q[indexHead]).ToList()
                })
                .ToArray();                
        }
        
        [Obsolete("Используйте методы расширения для моделей Sequential<T>")]
        /// <summary>
        /// Обучает модель. Поддерживает реккурентные сети.
        /// </summary>
        /// <param name="source"></param>
        /// <param name="trainDataSelector">Селектор, позволяющий указать для каждой эпохи свой набор данных для обучения.</param>
        /// <param name="lossFunction">Функция потерь</param>
        /// <param name="evaluationFunction">Оценочная функция</param>
        /// <param name="optimizer">Оптимизатор, используемый для обучения</param>
        /// <param name="epochCount">Количество эпох обучения</param>
        /// <param name="isReccurentModel">Указывает, что требуется обучать реккурентную модель</param>
        /// <param name="device">Устройство для обучения</param>
        /// <param name="ruleUpdateLearningRate">Правило обновления скорости обучения. Входные параметры: эпоха, текущая скорость обучения. Выходные: новая скорость обучения.</param>
        /// <param name="actionPerEpoch">Произвольное действие, которое требуется выполнять каждую эпоху. Позволяет прервать процесс тренировки. Входные параметры: эпоха, loss-ошибка, evaluation-ошибка. 
        /// Выходные: true - прервать процесс тренировки, false - продолжить тренировку.
        /// Используется для осуществления логирования, отображения процесса обучения, сохранения промежуточных чекпоинтов модели и т.п.</param>
        /// <returns></returns>
        public static FitResult Fit(this Function source,
           Func<int, IEnumerable<Minibatch>> trainDataSelector,
           Loss lossFunction,
           Loss evaluationFunction,
           Optimizer optimizer,
           int epochCount,
           bool isReccurentModel,
           DeviceDescriptor device,
           Func<int, double, double> ruleUpdateLearningRate = null,
           Func<int, double, double, bool> actionPerEpoch = null)
        {
            var inputVariable = source.Inputs.Single(p => p.Name.ToUpper() == "INPUT");
            var outputVariable = isReccurentModel ? Variable.InputVariable(source.Output.Shape, source.Output.DataType, "output", new List<Axis> { Axis.DefaultBatchAxis() })
                : Variable.InputVariable(source.Output.Shape, source.Output.DataType, "output");

            var loss = lossFunction.GetLoss(source, outputVariable, device);
            var evaluation = evaluationFunction.GetLoss(source, outputVariable, device);
            var learner = optimizer.GetOptimizer(source.Parameters());
            var trainer = CNTKLib.CreateTrainer(
                source,
                loss,
                evaluation,
                new LearnerVector() { learner });
            var learningRate = optimizer.LearningRate;
            var losses = new List<double>(epochCount);
            var evals = new List<double>(epochCount);
            Stopwatch sw = new Stopwatch();
            sw.Start();
            for (int i = 1; i <= epochCount; i++)
            {
                var trainMinibatches = trainDataSelector(i);
                foreach (var miniBatch in trainMinibatches)
                {
                    trainer.TrainMinibatch(new Dictionary<Variable, Value>() { { inputVariable, miniBatch.Features }, { outputVariable, miniBatch.Labels } }, false, device);
                }
                losses.Add(trainer.PreviousMinibatchLossAverage());
                evals.Add(trainer.PreviousMinibatchEvaluationAverage());

                bool needStopTraining = actionPerEpoch?.Invoke(i, losses[i - 1], evals[i - 1]) ?? false;
                if (needStopTraining)
                {
                    epochCount = i;
                    break;
                }

                if (ruleUpdateLearningRate != null)
                {
                    var proposaledLearningRate = ruleUpdateLearningRate(i, learningRate);
                    if (proposaledLearningRate != learningRate)
                    {
                        learner.SetLearningRateSchedule(new TrainingParameterScheduleDouble(learningRate));
                        learningRate = proposaledLearningRate;
                    }
                }
            }
            sw.Stop();

            return new FitResult()
            {
                LossError = losses[losses.Count - 1],
                EvaluationError = evals[evals.Count - 1],
                Duration = sw.Elapsed,
                EpochCount = epochCount,
                LossCurve = losses,
                EvaluationCurve = evals
            };
        }
        [Obsolete("Используйте методы расширения для моделей Sequential<T>")]
        /// <summary>
        /// Обучает модель. Поддерживает реккурентные сети.
        /// </summary>
        /// <param name="source"></param>
        /// <param name="trainData">Набор данных для обучения.</param>
        /// <param name="lossFunction">Функция потерь</param>
        /// <param name="evaluationFunction">Оценочная функция</param>
        /// <param name="optimizer">Оптимизатор, используемый для обучения</param>
        /// <param name="epochCount">Количество эпох обучения</param>
        /// <param name="isReccurentModel">Указывает, что требуется обучать реккурентную модель</param>
        /// <param name="device">Устройство для обучения</param>
        /// <param name="ruleUpdateLearningRate">Правило обновления скорости обучения. Входные параметры: эпоха, текущая скорость обучения. Выходные: новая скорость обучения.</param>
        /// <param name="actionPerEpoch">Произвольное действие, которое требуется выполнять каждую эпоху. Позволяет прервать процесс тренировки. Входные параметры: эпоха, loss-ошибка, evaluation-ошибка. 
        /// Выходные: true - прервать процесс тренировки, false - продолжить тренировку.
        /// Используется для осуществления логирования, отображения процесса обучения, сохранения промежуточных чекпоинтов модели и т.п.</param>
        /// <returns></returns>
        public static FitResult Fit(this Function source,
            IEnumerable<Minibatch> trainData,
            Loss lossFunction,
            Loss evaluationFunction,
            Optimizer optimizer,
            int epochCount,
            DeviceDescriptor device,
            bool isReccurentModel = false,
            Func<int, double, double> ruleUpdateLearningRate = null,
            Func<int, double, double, bool> actionPerEpoch = null)
        {
            return source.Fit(p => trainData, lossFunction, evaluationFunction, optimizer, epochCount, isReccurentModel, device, ruleUpdateLearningRate, actionPerEpoch);
        }
        [Obsolete("Используйте методы расширения для моделей Sequential<T>")]
        /// <summary>
        /// Обучает модель. Не применим для обучения реккуретных сетей.
        /// </summary>
        /// <typeparam name="T">Тип данных. Поддерживается <seealso cref="float"/>, <seealso cref="double"/></typeparam>
        /// <param name="source"></param>
        /// <param name="trainData">Набор данных для обучения. Каждый пример должен содержать в начале массива признаки размерностью inputDim, а в конце метки классов размерностью outputDim.
        /// Например inputDim = 3, outputDim = 2: [f1, f2, f3, l1, l2]</param>
        /// <param name="inputDim">Размерность признаков (разрядность)</param>
        /// <param name="minibatchSize">Размер минипакета</param>
        /// <param name="lossFunction">Функция потерь</param>
        /// <param name="evaluationFunction">Оценочная функция</param>
        /// <param name="optimizer">Оптимизатор, используемый для обучения</param>
        /// <param name="epochCount">Количество эпох обучения</param>        
        /// <param name="device">Устройство для обучения</param>
        /// <param name="shuffleSampleInMinibatchesPerEpoch">Указывает, что необходимо каждую эпоху перемешивать обучающие примеры для формирования новых минипакетов.</param>
        /// <param name="ruleUpdateLearningRate">Правило обновления скорости обучения. Входные параметры: эпоха, текущая скорость обучения. Выходные: новая скорость обучения.</param>
        /// <param name="actionPerEpoch">Произвольное действие, которое требуется выполнять каждую эпоху. Позволяет прервать процесс тренировки. Входные параметры: эпоха, loss-ошибка, evaluation-ошибка. 
        /// Выходные: true - прервать процесс тренировки, false - продолжить тренировку.
        /// Используется для осуществления логирования, отображения процесса обучения, сохранения промежуточных чекпоинтов модели и т.п.</param>
        /// <returns></returns>
        public static FitResult Fit<T>(this Function source,
            IList<T[]> trainData,
            int inputDim,
            int minibatchSize,
            Loss lossFunction,
            Loss evaluationFunction,
            Optimizer optimizer,
            int epochCount,
            DeviceDescriptor device,
            bool shuffleSampleInMinibatchesPerEpoch,
            Func<int, double, double> ruleUpdateLearningRate = null,
            Func<int, double, double, bool> actionPerEpoch = null) where T : IConvertible
        {
            DataConverter valueConverter = new DataConverter(device);
            IList<Minibatch> minibatches = null;
            if (!shuffleSampleInMinibatchesPerEpoch)
            {
                minibatches = valueConverter.ConvertDatasetToMinibatch(trainData, inputDim, minibatchSize).ToArray();
            }
            Func<int, IEnumerable<Minibatch>> getMinibatches = epoch =>
            {
                if (shuffleSampleInMinibatchesPerEpoch)
                {
                    trainData.Shuffle();
                    minibatches = valueConverter.ConvertDatasetToMinibatch(trainData, inputDim, minibatchSize).ToArray();
                }
                return minibatches;
            };
            return source.Fit(getMinibatches, lossFunction, evaluationFunction, optimizer, epochCount, false, device, ruleUpdateLearningRate, actionPerEpoch);
        }
        [Obsolete("Используйте методы расширения для моделей Sequential<T>")]
        /// <summary>
        /// Обучает реккурентную модель.
        /// </summary>
        /// <typeparam name="T">Тип данных. Поддерживается <seealso cref="float"/>, <seealso cref="double"/></typeparam>
        /// <param name="source"></param>
        /// <param name="features">Набор последовательностей (признаков). Каждая последовательность может быть переменной длинны, но одинаковой размерности (массивы из которых состоит последовательность, должны иметь одинаковую длину)</param>
        /// <param name="labels">Набор меток. Размерность меток должна быть одинаковая.</param>
        /// <param name="minibatchSize">Размер минипакета</param>
        /// <param name="lossFunction">Функция потерь</param>
        /// <param name="evaluationFunction">Оценочная функция</param>
        /// <param name="optimizer">Оптимизатор, используемый для обучения</param>
        /// <param name="epochCount">Количество эпох обучения</param>        
        /// <param name="device">Устройство для обучения</param>
        /// <param name="shuffleSampleInMinibatchesPerEpoch">Указывает, что необходимо каждую эпоху перемешивать обучающие примеры для формирования новых минипакетов.</param>
        /// <param name="ruleUpdateLearningRate">Правило обновления скорости обучения. Входные параметры: эпоха, текущая скорость обучения. Выходные: новая скорость обучения.</param>
        /// <param name="actionPerEpoch">Произвольное действие, которое требуется выполнять каждую эпоху. Позволяет прервать процесс тренировки. Входные параметры: эпоха, loss-ошибка, evaluation-ошибка. 
        /// Выходные: true - прервать процесс тренировки, false - продолжить тренировку.
        /// Используется для осуществления логирования, отображения процесса обучения, сохранения промежуточных чекпоинтов модели и т.п.</param>
        /// <returns></returns>
        public static FitResult Fit<T>(this Function source,
            IList<IList<T[]>> features,
            IList<T[]> labels,
            int minibatchSize,
            Loss lossFunction,
            Loss evaluationFunction,
            Optimizer optimizer,
            int epochCount,
            DeviceDescriptor device,
            bool shuffleSampleInMinibatchesPerEpoch,
            Func<int, double, double> ruleUpdateLearningRate = null,
            Func<int, double, double, bool> actionPerEpoch = null) where T : IConvertible
        {
            if (features.Count != labels.Count) throw new ArgumentException("Количество поледовательностей(features) и меток(labels) должно быть одинаковым.");

            DataConverter valueConverter = new DataConverter(device);
            IList<Minibatch> minibatches = null;
            if (!shuffleSampleInMinibatchesPerEpoch)
            {
                minibatches = valueConverter.ConvertDatasetToMinibatch(features, labels, minibatchSize).ToArray();
            }
            var trainData = features.Zip(labels, (f, l) => (f, l)).ToArray();
            Func<int, IEnumerable<Minibatch>> getMinibatches = epoch =>
            {
                if (shuffleSampleInMinibatchesPerEpoch)
                {
                    trainData.Shuffle();
                    minibatches = valueConverter.ConvertDatasetToMinibatch(trainData.Select(p => p.f), trainData.Select(p => p.l), minibatchSize).ToArray();
                }
                return minibatches;
            };

            return source.Fit(getMinibatches, lossFunction, evaluationFunction, optimizer, epochCount, true, device, ruleUpdateLearningRate, actionPerEpoch);
        }
        [Obsolete("Используйте методы расширения для моделей Sequential<T>")]
        /// <summary>
        /// Обучает модель с двумерным входом. Не применим для обучения реккуретных сетей.
        /// </summary>
        /// <typeparam name="T">Тип данных. Поддерживается <seealso cref="float"/>, <seealso cref="double"/></typeparam>
        /// <param name="source"></param>
        /// <param name="features">Набор данных для обучения.</param>
        /// <param name="minibatchSize">Размер минипакета</param>
        /// <param name="lossFunction">Функция потерь</param>
        /// <param name="evaluationFunction">Оценочная функция</param>
        /// <param name="optimizer">Оптимизатор, используемый для обучения</param>
        /// <param name="epochCount">Количество эпох обучения</param>        
        /// <param name="device">Устройство для обучения</param>
        /// <param name="shuffleSampleInMinibatchesPerEpoch">Указывает, что необходимо каждую эпоху перемешивать обучающие примеры для формирования новых минипакетов.</param>
        /// <param name="ruleUpdateLearningRate">Правило обновления скорости обучения. Входные параметры: эпоха, текущая скорость обучения. Выходные: новая скорость обучения.</param>
        /// <param name="actionPerEpoch">Произвольное действие, которое требуется выполнять каждую эпоху. Позволяет прервать процесс тренировки. Входные параметры: эпоха, loss-ошибка, evaluation-ошибка. 
        /// Выходные: true - прервать процесс тренировки, false - продолжить тренировку.
        /// Используется для осуществления логирования, отображения процесса обучения, сохранения промежуточных чекпоинтов модели и т.п.</param>
        /// <returns></returns>
        public static FitResult Fit<T>(this Function source,
            IList<T[,]> features,            
            IList<T[]> labels,
            int minibatchSize,
            Loss lossFunction,
            Loss evaluationFunction,
            Optimizer optimizer,
            int epochCount,
            DeviceDescriptor device,
            bool shuffleSampleInMinibatchesPerEpoch,
            Func<int, double, double> ruleUpdateLearningRate = null,
            Func<int, double, double, bool> actionPerEpoch = null) where T : IConvertible
        {
            DataConverter valueConverter = new DataConverter(device);
            IList<Minibatch> minibatches = null;
            if (!shuffleSampleInMinibatchesPerEpoch)
            {
                minibatches = valueConverter.ConvertDatasetToMinibatch(features, labels, minibatchSize).ToArray();
            }
            var trainData = features.Zip(labels, (f, l) => (f, l)).ToArray();
            Func<int, IEnumerable<Minibatch>> getMinibatches = epoch =>
            {
                if (shuffleSampleInMinibatchesPerEpoch)
                {
                    trainData.Shuffle();
                    minibatches = valueConverter.ConvertDatasetToMinibatch(trainData.Select(p => p.f), trainData.Select(p => p.l), minibatchSize).ToArray();
                }
                return minibatches;
            };
            return source.Fit(getMinibatches, lossFunction, evaluationFunction, optimizer, epochCount, false, device, ruleUpdateLearningRate, actionPerEpoch);
        }
        [Obsolete("Используйте методы расширения для моделей Sequential<T>")]
        /// <summary>
        /// Обучает модель. Не применим для обучения реккуретных сетей.
        /// </summary>
        /// <typeparam name="T">Тип данных. Поддерживается <seealso cref="float"/>, <seealso cref="double"/></typeparam>
        /// <param name="source"></param>
        /// <param name="trainData">Набор данных для обучения. Каждый пример должен содержать в начале массива признаки размерностью inputDim, а в конце метки классов размерностью outputDim.
        /// Например inputDim = 3, outputDim = 2: [f1, f2, f3, l1, l2]</param>
        /// <param name="inputDim">Размерность признаков (разрядность)</param>
        /// <param name="minibatchSize">Размер минипакета</param>
        /// <param name="lossFunction">Функция потерь</param>
        /// <param name="evaluationFunction">Оценочная функция</param>
        /// <param name="optimizer">Оптимизатор, используемый для обучения</param>
        /// <param name="epochCount">Количество эпох обучения</param>        
        /// <param name="device">Устройство для обучения</param>       
        /// <param name="ruleUpdateLearningRate">Правило обновления скорости обучения. Входные параметры: эпоха, текущая скорость обучения. Выходные: новая скорость обучения.</param>
        /// <param name="actionPerEpoch">Произвольное действие, которое требуется выполнять каждую эпоху. Позволяет прервать процесс тренировки. Входные параметры: эпоха, loss-ошибка, evaluation-ошибка. 
        /// Выходные: true - прервать процесс тренировки, false - продолжить тренировку.
        /// Используется для осуществления логирования, отображения процесса обучения, сохранения промежуточных чекпоинтов модели и т.п.</param>
        /// <returns></returns>
        public static FitResult Fit<T>(this Function source,
            IEnumerable<T[]> trainData,
            int inputDim,
            int minibatchSize,
            Loss lossFunction,
            Loss evaluationFunction,
            Optimizer optimizer,
            int epochCount,
            DeviceDescriptor device,
            Func<int, double, double> ruleUpdateLearningRate = null,
            Func<int, double, double, bool> actionPerEpoch = null) where T : IConvertible
        {
            DataConverter valueConverter = new DataConverter(device);
            var minibatches = valueConverter.ConvertDatasetToMinibatch(trainData, inputDim, minibatchSize);
            return source.Fit(p => minibatches, lossFunction, evaluationFunction, optimizer, epochCount, false, device, ruleUpdateLearningRate, actionPerEpoch);
        }
        [Obsolete("Используйте методы расширения для моделей Sequential<T>")]
        /// <summary>
        /// Обучает реккурентную модель.
        /// </summary>
        /// <typeparam name="T">Тип данных. Поддерживается <seealso cref="float"/>, <seealso cref="double"/></typeparam>
        /// <param name="source"></param>
        /// <param name="features">Набор последовательностей (признаков). Каждая последовательность может быть переменной длинны, но одинаковой размерности (массивы из которых состоит последовательность, должны иметь одинаковую длину)</param>
        /// <param name="labels">Набор меток. Размерность меток должна быть одинаковая.</param>
        /// <param name="minibatchSize">Размер минипакета</param>
        /// <param name="lossFunction">Функция потерь</param>
        /// <param name="evaluationFunction">Оценочная функция</param>
        /// <param name="optimizer">Оптимизатор, используемый для обучения</param>
        /// <param name="epochCount">Количество эпох обучения</param>        
        /// <param name="device">Устройство для обучения</param>     
        /// <param name="ruleUpdateLearningRate">Правило обновления скорости обучения. Входные параметры: эпоха, текущая скорость обучения. Выходные: новая скорость обучения.</param>
        /// <param name="actionPerEpoch">Произвольное действие, которое требуется выполнять каждую эпоху. Позволяет прервать процесс тренировки. Входные параметры: эпоха, loss-ошибка, evaluation-ошибка. 
        /// Выходные: true - прервать процесс тренировки, false - продолжить тренировку.
        /// Используется для осуществления логирования, отображения процесса обучения, сохранения промежуточных чекпоинтов модели и т.п.</param>
        /// <returns></returns>
        public static FitResult Fit<T>(this Function source,
            IEnumerable<IList<T[]>> features,
            IEnumerable<T[]> labels,
            int minibatchSize,
            Loss lossFunction,
            Loss evaluationFunction,
            Optimizer optimizer,
            int epochCount,
            DeviceDescriptor device,
            Func<int, double, double> ruleUpdateLearningRate = null,
            Func<int, double, double, bool> actionPerEpoch = null) where T : IConvertible
        {
            DataConverter valueConverter = new DataConverter(device);
            var minibatches = valueConverter.ConvertDatasetToMinibatch(features, labels, minibatchSize);
            return source.Fit(p => minibatches, lossFunction, evaluationFunction, optimizer, epochCount, true, device, ruleUpdateLearningRate, actionPerEpoch);
        }
        [Obsolete("Используйте методы расширения для моделей Sequential<T>")]
        /// <summary>
        /// Обучает модель с двумерным входом. Не применим для обучения реккуретных сетей.
        /// </summary>
        /// <typeparam name="T">Тип данных. Поддерживается <seealso cref="float"/>, <seealso cref="double"/></typeparam>
        /// <param name="source"></param>
        /// <param name="features">Набор данных для обучения.</param>
        /// <param name="minibatchSize">Размер минипакета</param>
        /// <param name="lossFunction">Функция потерь</param>
        /// <param name="evaluationFunction">Оценочная функция</param>
        /// <param name="optimizer">Оптимизатор, используемый для обучения</param>
        /// <param name="epochCount">Количество эпох обучения</param>        
        /// <param name="device">Устройство для обучения</param>  
        /// <param name="ruleUpdateLearningRate">Правило обновления скорости обучения. Входные параметры: эпоха, текущая скорость обучения. Выходные: новая скорость обучения.</param>
        /// <param name="actionPerEpoch">Произвольное действие, которое требуется выполнять каждую эпоху. Позволяет прервать процесс тренировки. Входные параметры: эпоха, loss-ошибка, evaluation-ошибка. 
        /// Выходные: true - прервать процесс тренировки, false - продолжить тренировку.
        /// Используется для осуществления логирования, отображения процесса обучения, сохранения промежуточных чекпоинтов модели и т.п.</param>
        /// <returns></returns>
        public static FitResult Fit<T>(this Function source,
            IEnumerable<T[,]> features,
            IEnumerable<T[]> labels,
            int minibatchSize,
            Loss lossFunction,
            Loss evaluationFunction,
            Optimizer optimizer,
            int epochCount,
            DeviceDescriptor device,
            Func<int, double, double> ruleUpdateLearningRate = null,
            Func<int, double, double, bool> actionPerEpoch = null) where T : IConvertible
        {
            DataConverter valueConverter = new DataConverter(device);
            var minibatches = valueConverter.ConvertDatasetToMinibatch(features, labels, minibatchSize);
            return source.Fit(p => minibatches, lossFunction, evaluationFunction, optimizer, epochCount, false, device, ruleUpdateLearningRate, actionPerEpoch);
        }
        #endregion

        #region Extensions for Sequential<T>
        /// <summary>
        /// Обучает модель. Поддерживает реккурентные сети.
        /// </summary>
        /// <param name="source"></param>
        /// <param name="trainDataSelector">Селектор, позволяющий указать для каждой эпохи свой набор данных для обучения.</param>
        /// <param name="lossFunction">Функция потерь</param>
        /// <param name="evaluationFunction">Оценочная функция</param>
        /// <param name="optimizer">Оптимизатор, используемый для обучения</param>
        /// <param name="epochCount">Количество эпох обучения</param>
        /// <param name="isReccurentModel">Указывает, что требуется обучать реккурентную модель</param>
        /// <param name="device">Устройство для обучения</param>
        /// <param name="ruleUpdateLearningRate">Правило обновления скорости обучения. Входные параметры: эпоха, текущая скорость обучения. Выходные: новая скорость обучения.</param>
        /// <param name="actionPerEpoch">Произвольное действие, которое требуется выполнять каждую эпоху. Позволяет прервать процесс тренировки. Входные параметры: эпоха, loss-ошибка, evaluation-ошибка. 
        /// Выходные: true - прервать процесс тренировки, false - продолжить тренировку.
        /// Используется для осуществления логирования, отображения процесса обучения, сохранения промежуточных чекпоинтов модели и т.п.</param>
        /// <returns></returns>
        public static FitResult Fit<T>(this Sequential<T> source,
           Func<int, IEnumerable<Minibatch>> trainDataSelector,
           Loss lossFunction,
           Loss evaluationFunction,
           Optimizer optimizer,
           int epochCount,
           bool isReccurentModel,
           DeviceDescriptor device,
           Func<int, double, double> ruleUpdateLearningRate = null,
           Func<int, double, double, bool> actionPerEpoch = null) where T : IConvertible
        {
            return source.Model.Fit(trainDataSelector, lossFunction, evaluationFunction, optimizer, epochCount, isReccurentModel, device, ruleUpdateLearningRate, actionPerEpoch);
        }
        /// <summary>
        /// Обучает модель. Поддерживает реккурентные сети.
        /// </summary>
        /// <param name="source"></param>
        /// <param name="trainData">Набор данных для обучения.</param>
        /// <param name="lossFunction">Функция потерь</param>
        /// <param name="evaluationFunction">Оценочная функция</param>
        /// <param name="optimizer">Оптимизатор, используемый для обучения</param>
        /// <param name="epochCount">Количество эпох обучения</param>
        /// <param name="isReccurentModel">Указывает, что требуется обучать реккурентную модель</param>
        /// <param name="device">Устройство для обучения</param>
        /// <param name="ruleUpdateLearningRate">Правило обновления скорости обучения. Входные параметры: эпоха, текущая скорость обучения. Выходные: новая скорость обучения.</param>
        /// <param name="actionPerEpoch">Произвольное действие, которое требуется выполнять каждую эпоху. Позволяет прервать процесс тренировки. Входные параметры: эпоха, loss-ошибка, evaluation-ошибка. 
        /// Выходные: true - прервать процесс тренировки, false - продолжить тренировку.
        /// Используется для осуществления логирования, отображения процесса обучения, сохранения промежуточных чекпоинтов модели и т.п.</param>
        /// <returns></returns>
        public static FitResult Fit<T>(this Sequential<T> source,
            IEnumerable<Minibatch> trainData,
            Loss lossFunction,
            Loss evaluationFunction,
            Optimizer optimizer,
            int epochCount,
            DeviceDescriptor device,
            bool isReccurentModel = false,
            Func<int, double, double> ruleUpdateLearningRate = null,
            Func<int, double, double, bool> actionPerEpoch = null) where T : IConvertible
        {
            return source.Model.Fit(p => trainData, lossFunction, evaluationFunction, optimizer, epochCount, isReccurentModel, device, ruleUpdateLearningRate, actionPerEpoch);
        }
        /// <summary>
        /// Обучает модель. Не применим для обучения реккуретных сетей.
        /// </summary>
        /// <typeparam name="T">Тип данных. Поддерживается <seealso cref="float"/>, <seealso cref="double"/></typeparam>
        /// <param name="source"></param>
        /// <param name="trainData">Набор данных для обучения. Каждый пример должен содержать в начале массива признаки размерностью inputDim, а в конце метки классов размерностью outputDim.
        /// Например inputDim = 3, outputDim = 2: [f1, f2, f3, l1, l2]</param>
        /// <param name="inputDim">Размерность признаков (разрядность)</param>
        /// <param name="minibatchSize">Размер минипакета</param>
        /// <param name="lossFunction">Функция потерь</param>
        /// <param name="evaluationFunction">Оценочная функция</param>
        /// <param name="optimizer">Оптимизатор, используемый для обучения</param>
        /// <param name="epochCount">Количество эпох обучения</param>        
        /// <param name="device">Устройство для обучения</param>
        /// <param name="shuffleSampleInMinibatchesPerEpoch">Указывает, что необходимо каждую эпоху перемешивать обучающие примеры для формирования новых минипакетов.</param>
        /// <param name="ruleUpdateLearningRate">Правило обновления скорости обучения. Входные параметры: эпоха, текущая скорость обучения. Выходные: новая скорость обучения.</param>
        /// <param name="actionPerEpoch">Произвольное действие, которое требуется выполнять каждую эпоху. Позволяет прервать процесс тренировки. Входные параметры: эпоха, loss-ошибка, evaluation-ошибка. 
        /// Выходные: true - прервать процесс тренировки, false - продолжить тренировку.
        /// Используется для осуществления логирования, отображения процесса обучения, сохранения промежуточных чекпоинтов модели и т.п.</param>
        /// <returns></returns>
        public static FitResult Fit<T>(this Sequential<T> source,
            IList<T[]> trainData,
            int inputDim,
            int minibatchSize,
            Loss lossFunction,
            Loss evaluationFunction,
            Optimizer optimizer,
            int epochCount,
            DeviceDescriptor device,
            bool shuffleSampleInMinibatchesPerEpoch,
            Func<int, double, double> ruleUpdateLearningRate = null,
            Func<int, double, double, bool> actionPerEpoch = null) where T : IConvertible
        {
            return source.Model.Fit(trainData, inputDim, minibatchSize, lossFunction, evaluationFunction, optimizer, epochCount, device, shuffleSampleInMinibatchesPerEpoch, ruleUpdateLearningRate, actionPerEpoch);
        }
        /// <summary>
        /// Обучает реккурентную модель.
        /// </summary>
        /// <typeparam name="T">Тип данных. Поддерживается <seealso cref="float"/>, <seealso cref="double"/></typeparam>
        /// <param name="source"></param>
        /// <param name="features">Набор последовательностей (признаков). Каждая последовательность может быть переменной длинны, но одинаковой размерности (массивы из которых состоит последовательность, должны иметь одинаковую длину)</param>
        /// <param name="labels">Набор меток. Размерность меток должна быть одинаковая.</param>
        /// <param name="minibatchSize">Размер минипакета</param>
        /// <param name="lossFunction">Функция потерь</param>
        /// <param name="evaluationFunction">Оценочная функция</param>
        /// <param name="optimizer">Оптимизатор, используемый для обучения</param>
        /// <param name="epochCount">Количество эпох обучения</param>        
        /// <param name="device">Устройство для обучения</param>
        /// <param name="shuffleSampleInMinibatchesPerEpoch">Указывает, что необходимо каждую эпоху перемешивать обучающие примеры для формирования новых минипакетов.</param>
        /// <param name="ruleUpdateLearningRate">Правило обновления скорости обучения. Входные параметры: эпоха, текущая скорость обучения. Выходные: новая скорость обучения.</param>
        /// <param name="actionPerEpoch">Произвольное действие, которое требуется выполнять каждую эпоху. Позволяет прервать процесс тренировки. Входные параметры: эпоха, loss-ошибка, evaluation-ошибка. 
        /// Выходные: true - прервать процесс тренировки, false - продолжить тренировку.
        /// Используется для осуществления логирования, отображения процесса обучения, сохранения промежуточных чекпоинтов модели и т.п.</param>
        /// <returns></returns>
        public static FitResult Fit<T>(this Sequential<T> source,
            IList<IList<T[]>> features,
            IList<T[]> labels,
            int minibatchSize,
            Loss lossFunction,
            Loss evaluationFunction,
            Optimizer optimizer,
            int epochCount,
            DeviceDescriptor device,
            bool shuffleSampleInMinibatchesPerEpoch,
            Func<int, double, double> ruleUpdateLearningRate = null,
            Func<int, double, double, bool> actionPerEpoch = null) where T : IConvertible
        {
            return source.Model.Fit(features, labels, minibatchSize, lossFunction, evaluationFunction, optimizer, epochCount, device, shuffleSampleInMinibatchesPerEpoch, ruleUpdateLearningRate, actionPerEpoch);
        }
        /// <summary>
        /// Обучает модель с двумерным входом. Не применим для обучения реккуретных сетей.
        /// </summary>
        /// <typeparam name="T">Тип данных. Поддерживается <seealso cref="float"/>, <seealso cref="double"/></typeparam>
        /// <param name="source"></param>
        /// <param name="features">Набор данных для обучения.</param>
        /// <param name="minibatchSize">Размер минипакета</param>
        /// <param name="lossFunction">Функция потерь</param>
        /// <param name="evaluationFunction">Оценочная функция</param>
        /// <param name="optimizer">Оптимизатор, используемый для обучения</param>
        /// <param name="epochCount">Количество эпох обучения</param>        
        /// <param name="device">Устройство для обучения</param>  
        /// <param name="shuffleSampleInMinibatchesPerEpoch">Указывает, что необходимо каждую эпоху перемешивать обучающие примеры для формирования новых минипакетов.</param>
        /// <param name="ruleUpdateLearningRate">Правило обновления скорости обучения. Входные параметры: эпоха, текущая скорость обучения. Выходные: новая скорость обучения.</param>
        /// <param name="actionPerEpoch">Произвольное действие, которое требуется выполнять каждую эпоху. Позволяет прервать процесс тренировки. Входные параметры: эпоха, loss-ошибка, evaluation-ошибка. 
        /// Выходные: true - прервать процесс тренировки, false - продолжить тренировку.
        /// Используется для осуществления логирования, отображения процесса обучения, сохранения промежуточных чекпоинтов модели и т.п.</param>
        /// <returns></returns>
        public static FitResult Fit<T>(this Sequential<T> source,
            IList<T[,]> features,
            IList<T[]> labels,
            int minibatchSize,
            Loss lossFunction,
            Loss evaluationFunction,
            Optimizer optimizer,
            int epochCount,
            DeviceDescriptor device,
            bool shuffleSampleInMinibatchesPerEpoch,
            Func<int, double, double> ruleUpdateLearningRate = null,
            Func<int, double, double, bool> actionPerEpoch = null) where T : IConvertible
        {
            return source.Model.Fit(features, labels, minibatchSize, lossFunction, evaluationFunction, optimizer, epochCount, device, shuffleSampleInMinibatchesPerEpoch, ruleUpdateLearningRate, actionPerEpoch);
        }
        /// <summary>
        /// Обучает модель. Не применим для обучения реккуретных сетей.
        /// </summary>
        /// <typeparam name="T">Тип данных. Поддерживается <seealso cref="float"/>, <seealso cref="double"/></typeparam>
        /// <param name="source"></param>
        /// <param name="trainData">Набор данных для обучения. Каждый пример должен содержать в начале массива признаки размерностью inputDim, а в конце метки классов размерностью outputDim.
        /// Например inputDim = 3, outputDim = 2: [f1, f2, f3, l1, l2]</param>
        /// <param name="inputDim">Размерность признаков (разрядность)</param>
        /// <param name="minibatchSize">Размер минипакета</param>
        /// <param name="lossFunction">Функция потерь</param>
        /// <param name="evaluationFunction">Оценочная функция</param>
        /// <param name="optimizer">Оптимизатор, используемый для обучения</param>
        /// <param name="epochCount">Количество эпох обучения</param>        
        /// <param name="device">Устройство для обучения</param>       
        /// <param name="ruleUpdateLearningRate">Правило обновления скорости обучения. Входные параметры: эпоха, текущая скорость обучения. Выходные: новая скорость обучения.</param>
        /// <param name="actionPerEpoch">Произвольное действие, которое требуется выполнять каждую эпоху. Позволяет прервать процесс тренировки. Входные параметры: эпоха, loss-ошибка, evaluation-ошибка. 
        /// Выходные: true - прервать процесс тренировки, false - продолжить тренировку.
        /// Используется для осуществления логирования, отображения процесса обучения, сохранения промежуточных чекпоинтов модели и т.п.</param>
        /// <returns></returns>
        public static FitResult Fit<T>(this Sequential<T> source,
            IEnumerable<T[]> trainData,
            int inputDim,
            int minibatchSize,
            Loss lossFunction,
            Loss evaluationFunction,
            Optimizer optimizer,
            int epochCount,
            DeviceDescriptor device,
            Func<int, double, double> ruleUpdateLearningRate = null,
            Func<int, double, double, bool> actionPerEpoch = null) where T : IConvertible
        {
            return source.Model.Fit(trainData, inputDim, minibatchSize, lossFunction, evaluationFunction, optimizer, epochCount, device, ruleUpdateLearningRate, actionPerEpoch);
        }
        /// <summary>
        /// Обучает реккурентную модель.
        /// </summary>
        /// <typeparam name="T">Тип данных. Поддерживается <seealso cref="float"/>, <seealso cref="double"/></typeparam>
        /// <param name="source"></param>
        /// <param name="features">Набор последовательностей (признаков). Каждая последовательность может быть переменной длинны, но одинаковой размерности (массивы из которых состоит последовательность, должны иметь одинаковую длину)</param>
        /// <param name="labels">Набор меток. Размерность меток должна быть одинаковая.</param>
        /// <param name="minibatchSize">Размер минипакета</param>
        /// <param name="lossFunction">Функция потерь</param>
        /// <param name="evaluationFunction">Оценочная функция</param>
        /// <param name="optimizer">Оптимизатор, используемый для обучения</param>
        /// <param name="epochCount">Количество эпох обучения</param>        
        /// <param name="device">Устройство для обучения</param>        
        /// <param name="ruleUpdateLearningRate">Правило обновления скорости обучения. Входные параметры: эпоха, текущая скорость обучения. Выходные: новая скорость обучения.</param>
        /// <param name="actionPerEpoch">Произвольное действие, которое требуется выполнять каждую эпоху. Позволяет прервать процесс тренировки. Входные параметры: эпоха, loss-ошибка, evaluation-ошибка. 
        /// Выходные: true - прервать процесс тренировки, false - продолжить тренировку.
        /// Используется для осуществления логирования, отображения процесса обучения, сохранения промежуточных чекпоинтов модели и т.п.</param>
        /// <returns></returns>
        public static FitResult Fit<T>(this Sequential<T> source,
            IEnumerable<IList<T[]>> features,
            IEnumerable<T[]> labels,
            int minibatchSize,
            Loss lossFunction,
            Loss evaluationFunction,
            Optimizer optimizer,
            int epochCount,
            DeviceDescriptor device,
            Func<int, double, double> ruleUpdateLearningRate = null,
            Func<int, double, double, bool> actionPerEpoch = null) where T : IConvertible
        {
            return source.Model.Fit(features, labels, minibatchSize, lossFunction, evaluationFunction, optimizer, epochCount, device, ruleUpdateLearningRate, actionPerEpoch);
        }
        /// <summary>
        /// Обучает модель с двумерным входом. Не применим для обучения реккуретных сетей.
        /// </summary>
        /// <typeparam name="T">Тип данных. Поддерживается <seealso cref="float"/>, <seealso cref="double"/></typeparam>
        /// <param name="source"></param>
        /// <param name="features">Набор данных для обучения.</param>
        /// <param name="minibatchSize">Размер минипакета</param>
        /// <param name="lossFunction">Функция потерь</param>
        /// <param name="evaluationFunction">Оценочная функция</param>
        /// <param name="optimizer">Оптимизатор, используемый для обучения</param>
        /// <param name="epochCount">Количество эпох обучения</param>        
        /// <param name="device">Устройство для обучения</param>          
        /// <param name="ruleUpdateLearningRate">Правило обновления скорости обучения. Входные параметры: эпоха, текущая скорость обучения. Выходные: новая скорость обучения.</param>
        /// <param name="actionPerEpoch">Произвольное действие, которое требуется выполнять каждую эпоху. Позволяет прервать процесс тренировки. Входные параметры: эпоха, loss-ошибка, evaluation-ошибка. 
        /// Выходные: true - прервать процесс тренировки, false - продолжить тренировку.
        /// Используется для осуществления логирования, отображения процесса обучения, сохранения промежуточных чекпоинтов модели и т.п.</param>
        /// <returns></returns>
        public static FitResult Fit<T>(this Sequential<T> source,
            IEnumerable<T[,]> features,
            IEnumerable<T[]> labels,
            int minibatchSize,
            Loss lossFunction,
            Loss evaluationFunction,
            Optimizer optimizer,
            int epochCount,
            DeviceDescriptor device,
            Func<int, double, double> ruleUpdateLearningRate = null,
            Func<int, double, double, bool> actionPerEpoch = null) where T : IConvertible
        {
            return source.Model.Fit(features, labels, minibatchSize, lossFunction, evaluationFunction, optimizer, epochCount, device, ruleUpdateLearningRate, actionPerEpoch);
        }
        #endregion

    }
}


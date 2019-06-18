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
using EasyCNTK.Learning.Metrics;
using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;

namespace EasyCNTK.Learning
{
    public static class EvaluateExtensions
    {
        /// <summary>
        /// Возвращает метрики для задач регрессии. Если целевая переменная многомерная, метрики возвращаются для каждого измерения независимо.
        /// </summary>
        /// <typeparam name="T">Тип данных. Поддерживается <seealso cref="float"/>, <seealso cref="double"/></typeparam>
        /// <param name="source"></param>
        /// <returns></returns>
        public static IList<RegressionMetrics> GetRegressionMetrics<T>(this IEnumerable<EvaluateItem<T>> source) where T: IConvertible
        {
            var firstItem = source.FirstOrDefault();
            if (firstItem.Equals(default(EvaluateItem<T>)))
            {
                throw new ArgumentException("Последовательность IEnumerable<EvaluateItem<T>> не содержит элементов.", "source");
            }
            
            var result = firstItem.EvaluatedValue
                .Select(p => new RegressionMetrics())
                .ToArray();
           
            var expectedDataAccumulator = new double[result.Length];
            int countItems = 0;
            foreach (var item in source)
            {
                for (int i = 0; i < result.Length; i++)
                {
                    double evaluated = item.EvaluatedValue[i].ToDouble(CultureInfo.InvariantCulture);
                    double expected  = item.ExpectedValue[i].ToDouble(CultureInfo.InvariantCulture);
                    double mae       = Math.Abs(evaluated - expected);
                    double rmse      = Math.Pow(evaluated - expected, 2);
                    checked
                    {
                        result[i].MAE += mae;
                        result[i].RMSE += rmse;                        
                        expectedDataAccumulator[i] += expected;
                    }
                }

                countItems++;
            }
            for (int i = 0; i < result.Length; i++)
            {
                expectedDataAccumulator[i] = expectedDataAccumulator[i] / countItems;
            }

            var expectedVarianceAccumulator = new double[result.Length];
            foreach (var item in source)
            {
                for (int i = 0; i < result.Length; i++)
                {
                    double expected = item.ExpectedValue[i].ToDouble(CultureInfo.InvariantCulture);
                    checked
                    {
                        expectedVarianceAccumulator[i] += Math.Pow(expected - expectedDataAccumulator[i], 2);
                    }
                }
            }

            for (int i = 0; i < result.Length; i++)
            {
                result[i].Determination = 1 - (result[i].RMSE / expectedVarianceAccumulator[i]);
                result[i].MAE = result[i].MAE / countItems;
                result[i].RMSE = Math.Sqrt(result[i].RMSE / countItems);
                
            }

            return result;
        }
        /// <summary>
        /// Вычисляет метрики для задач бинарной классификации. 
        /// Подразумевается, что выход имеет единичную размерность и метки классов закодированы в 1 для True наблюдений, в 0 для False.
        /// </summary>
        /// <typeparam name="T">Тип данных. Поддерживается <seealso cref="float"/>, <seealso cref="double"/></typeparam>
        /// <param name="source"></param>
        /// <param name="threshold">Пороговое значение для действительного значения выхода нейросети, ниже которого класс определяется как False. </param>
        /// /// <returns></returns>
        public static BinaryClassificationMetrics GetBinaryClassificationMetrics<T>(this IEnumerable<EvaluateItem<T>> source, double threshold = 0.5) where T: IConvertible
        {
            var firstItem = source.FirstOrDefault();
            if (firstItem.Equals(default(EvaluateItem<T>)))
            {
                throw new ArgumentException("Последовательность IEnumerable<EvaluateItem<T>> не содержит элементов.", "source");
            }

            int TP = 0; //факт 1, оценка 1
            int TN = 0; //факт 0, оценка 0
            int FP = 0; //факт 0, оценка 1
            int FN = 0; //факт 1, оценка 0

            foreach (var item in source)
            {                
                int expected = item.ExpectedValue[0].ToInt32(CultureInfo.InvariantCulture);
                int evaluated = item.EvaluatedValue[0].ToDouble(CultureInfo.InvariantCulture) < threshold ? 0 : 1;

                bool isPositive = expected == 1;
                if (isPositive)
                {
                    if (expected == evaluated)
                    {
                        TP++; 
                    }
                    else
                    {
                        FN++;
                    }
                }
                else
                {
                    if (expected == evaluated)
                    {
                        TN++;
                    }
                    else
                    {
                        FP++;
                    }
                }
            }

            double countSamples = TP + TN + FP + FN;
            var accuracy = (TP + TN) / countSamples;
            var precision = (double)TP / (TP + FP);
            var recall = (double)TP / (TP + FN);
            var f1score = 2 * precision * recall / (precision + recall);
            var confusionMatrix = new double[2, 2]
            {
                { TP/countSamples, FP/countSamples },
                { FN/countSamples, TN/countSamples }
            };

            return new BinaryClassificationMetrics()
            {
                Accuracy = accuracy,
                Precision = precision,
                Recall = recall,
                F1Score = f1score,
                ConfusionMatix = confusionMatrix
            };
        }
        /// <summary>
        /// Вычисляет метрики для задач одноклассовой классификации. 
        /// Подразумевается, что выход закодирован в One-Hot-Encoding(и обернут в Softmax, хотя возможно использовать <seealso cref="ActivationFunctions.Sigmoid"/>, <seealso cref="ActivationFunctions.HardSigmoid"/>), в ином случае метрика рассчитается некорректно.
        /// </summary>
        /// <typeparam name="T">Тип данных. Поддерживается <seealso cref="float"/>, <seealso cref="double"/></typeparam>
        /// <param name="source"></param>
        /// <returns></returns>
        public static OneLabelClassificationMetrics GetOneLabelClassificationMetrics<T>(this IEnumerable<EvaluateItem<T>> source) where T:IConvertible
        {
            var firstElement = source.FirstOrDefault();
            if (firstElement.Equals(default(EvaluateItem<T>)))
            {
                throw new ArgumentException("Последовательность IEnumerable<EvaluateItem<T>> не содержит элементов.", "source");
            }
            
            var confusionMatrix = new double[firstElement.EvaluatedValue.Count, firstElement.EvaluatedValue.Count];
            var classesDistribution = Enumerable.Range(0, firstElement.EvaluatedValue.Count)
                .Select(p => new ClassItem()
                {
                    Index = p
                })
                .ToList();
            int countAccurateSamples = 0;
            int countSamples = 0;
            foreach (var item in source)
            {
                int expected = item.ExpectedValue.IndexOf(item.ExpectedValue.Max());
                int evaluated = item.EvaluatedValue.IndexOf(item.EvaluatedValue.Max());

                classesDistribution[expected].Fraction++;
                confusionMatrix[expected, evaluated]++;
                if (expected == evaluated)
                {
                    countAccurateSamples++;                    
                    classesDistribution[evaluated].Recall++;
                }
                classesDistribution[evaluated].Precision++;
                countSamples++;
            }
            for (int i = 0; i < firstElement.EvaluatedValue.Count; i++)
            {
                classesDistribution[i].Precision = classesDistribution[i].Precision == 0 ? 0 : classesDistribution[i].Recall / classesDistribution[i].Precision;
                classesDistribution[i].Recall /= classesDistribution[i].Fraction;
                classesDistribution[i].F1Score = (classesDistribution[i].Precision + classesDistribution[i].Recall) == 0 ? 0
                    : 2 * classesDistribution[i].Precision * classesDistribution[i].Recall / (classesDistribution[i].Precision + classesDistribution[i].Recall);
                classesDistribution[i].Fraction /= countSamples;
                for (int j = 0; j < firstElement.EvaluatedValue.Count; j++)
                {
                    confusionMatrix[i, j] /= countSamples;
                }
            }
                
            double accuracy = (double)countAccurateSamples / countSamples;
            return new OneLabelClassificationMetrics()
            {
                Accuracy = accuracy,
                ConfusionMatrix = confusionMatrix,
                ClassesDistribution = classesDistribution
            };
        }
        /// <summary>
        /// Вычисляет метрики для задач многоклассовой классификации.
        /// Подразумевается, что выход закодирован в One-Hot-Encoding(и обернут в <seealso cref="ActivationFunctions.Sigmoid"/>, <seealso cref="ActivationFunctions.HardSigmoid"/> и т.п.), в ином случае метрика рассчитается некорректно.
        /// </summary>
        /// <typeparam name="T">Тип данных. Поддерживается <seealso cref="float"/>, <seealso cref="double"/></typeparam>
        /// <param name="source"></param>
        /// <param name="thershold">Пороговое значение для действительного значения выхода нейросети, ниже которого класс не распознается. Другими словами - это минимальная вероятность, которую должен выдать классификатор для конкретного класса, чтобы этот класс был учтен как распознанный.</param>
        /// <returns></returns>
        public static MultiLabelClassificationMetrics GetMultiLabelClassificationMetrics<T>(this IEnumerable<EvaluateItem<T>> source, double thershold = 0.5) where T:IConvertible
        {
            var firstElement = source.FirstOrDefault();
            if (firstElement.Equals(default(EvaluateItem<T>)))
            {
                throw new ArgumentException("Последовательность IEnumerable<EvaluateItem<T>> не содержит элементов.", "source");
            }
            
            var classesDistribution = Enumerable.Range(0, firstElement.EvaluatedValue.Count)
                .Select(p => new ClassItem()
                {
                    Index = p
                })
                .ToList();
            int countAccurateLabels = 0;
            int countLabels = 0;
            foreach (var item in source)
            {
                var expected = item.ExpectedValue
                    .Select((value, index) => value.ToDouble(CultureInfo.InvariantCulture) > thershold ? index : -1)
                    .Where(p => p != -1)
                    .ToList();
                var evaluated = item.EvaluatedValue
                    .Select((value, index) => value.ToDouble(CultureInfo.InvariantCulture) > thershold ? index : -1)
                    .Where(p => p != -1)
                    .ToList();

                foreach (var target in expected)
                {
                    classesDistribution[target].Fraction++;
                    if (evaluated.Contains(target))
                    {
                        classesDistribution[target].Recall++;
                    }
                    countLabels++;
                }
                evaluated.ForEach(evaluate => classesDistribution[evaluate].Precision++);
            }
            classesDistribution.ForEach(p =>
            {
                p.Precision = p.Precision == 0 ? 0 : p.Recall / p.Precision;
                p.Recall /= p.Fraction;
                p.F1Score = (p.Precision + p.Recall) == 0 ? 0 : 2 * p.Precision * p.Recall / (p.Precision + p.Recall);
                p.Fraction /= countLabels;
            });

            double accuracy = (double)countAccurateLabels / countLabels;
            return new MultiLabelClassificationMetrics()
            {
                Accuracy = accuracy,                
                ClassesDistribution = classesDistribution
            };
        }


        #region Function extensions

        #region Evaluate<T>
        /// <summary>
        /// Вычисляет выход модели для каждого из тестовых примеров.
        /// </summary>
        /// <typeparam name="T">Тип данных. Поддерживается <seealso cref="float"/>, <seealso cref="double"/></typeparam>
        /// <param name="source"></param>
        /// <param name="testData">Тестовые данные. Каждый минипакет должен содержать 1 тестовый пример.</param>
        /// <param name="device">Устройство для расчетов</param>
        /// <returns></returns>
        public static IEnumerable<EvaluateItem<T>> Evaluate<T>(this Function source,
            IEnumerable<Minibatch> testData,
            DeviceDescriptor device) where T : IConvertible
        {
            var inputVariable = source.Inputs.Single(p => p.Name.ToUpper() == "INPUT");
            foreach (var miniBatch in testData)
            {
                var inputDataMap = new Dictionary<Variable, Value>() { { inputVariable, miniBatch.Features } };
                var outputDataMap = new Dictionary<Variable, Value>() { { source.Output, null } };

                source.Evaluate(inputDataMap, outputDataMap, device);

                var expected = miniBatch.Labels.GetDenseData<T>(source.Output);
                var evaluated = outputDataMap[source.Output].GetDenseData<T>(source.Output);

                foreach (var item in expected.Zip(evaluated, (exp, eval) => (exp, eval)))
                {
                    var evaluateItem = new EvaluateItem<T>(item.exp, item.eval);
                    yield return evaluateItem;
                }
            }
        }
        /// <summary>
        /// Вычисляет выход модели для каждого из тестовых примеров.
        /// </summary>
        /// <typeparam name="T">Тип данных. Поддерживается <seealso cref="float"/>, <seealso cref="double"/></typeparam>
        /// <param name="source"></param>
        /// <param name="testData">Набор тестовых данных для обучения. Каждый пример должен содержать в начале массива признаки размерностью inputDim, а в конце метки классов размерностью outputDim.
        /// Например inputDim = 3, outputDim = 2: [f1, f2, f3, l1, l2]</param>
        /// <param name="inputDim">Размерность признаков (разрядность)</param>
        /// <param name="device">Устройство для расчетов</param>
        /// <param name="minibatchSize">Размер минипакета для оценки. Использование позволяет оценивать данные пачками(параллельно), не тратя ресурсы на пересылку данных в память. Оптимальный размер зависит от объема данных, доступной памяти GPU (лучшее ускорение).</param>
        /// <returns></returns>
        public static IEnumerable<EvaluateItem<T>> Evaluate<T>(this Function source,
            IEnumerable<T[]> testData,
            int inputDim,
            DeviceDescriptor device,
            int minibatchSize = 512) where T : IConvertible
        {
            DataConverter valueConverter = new DataConverter(device);
            var test = valueConverter.ConvertDatasetToMinibatch(testData, inputDim, minibatchSize);
            return source.Evaluate<T>(test, device);
        }
        /// <summary>
        /// Вычисляет выход модели для каждого из тестовых примеров.
        /// </summary>
        /// <typeparam name="T">Тип данных. Поддерживается <seealso cref="float"/>, <seealso cref="double"/></typeparam>
        /// <param name="source"></param>
        /// <param name="features">Набор тестовых последовательностей (признаков). Каждая последовательность может быть переменной длинны, но одинаковой размерности (массивы из которых состоит последовательность, должны иметь одинаковую длину)</param>
        /// <param name="labels">Набор тестовых меток. Размерность меток должна быть одинаковая.</param>
        /// <param name="device">Устройство для расчетов</param>
        /// <param name="minibatchSize">Размер минипакета для оценки. Использование позволяет оценивать данные пачками(параллельно), не тратя ресурсы на пересылку данных в память. Оптимальный размер зависит от объема данных, доступной памяти GPU (лучшее ускорение).</param>
        /// <returns></returns>
        public static IEnumerable<EvaluateItem<T>> Evaluate<T>(this Function source,
            IEnumerable<IList<T[]>> features,
            IEnumerable<T[]> labels,
            DeviceDescriptor device,
            int minibatchSize = 512) where T : IConvertible
        {
            DataConverter valueConverter = new DataConverter(device);
            var test = valueConverter.ConvertDatasetToMinibatch(features, labels, minibatchSize);
            return source.Evaluate<T>(test, device);
        }
        /// <summary>
        /// Вычисляет выход модели для каждого из тестовых примеров.
        /// </summary>
        /// <typeparam name="T">Тип данных. Поддерживается <seealso cref="float"/>, <seealso cref="double"/></typeparam>
        /// <param name="source"></param>
        /// <param name="features">Набор тестовых данных</param>
        /// <param name="device">Устройство для расчетов</param>
        /// <param name="minibatchSize">Размер минипакета для оценки. Использование позволяет оценивать данные пачками(параллельно), не тратя ресурсы на пересылку данных в память. Оптимальный размер зависит от объема данных, доступной памяти GPU (лучшее ускорение).</param>
        /// <returns></returns>
        public static IEnumerable<EvaluateItem<T>> Evaluate<T>(this Function source,
            IEnumerable<T[,]> features,
            IEnumerable<T[]> labels,
            DeviceDescriptor device,
            int minibatchSize = 512) where T : IConvertible
        {
            DataConverter valueConverter = new DataConverter(device);
            var test = valueConverter.ConvertDatasetToMinibatch(features, labels, minibatchSize);
            return source.Evaluate<T>(test, device);
        }
        #endregion

        #region Predict<T>
        /// <summary>
        /// Вычисляет выходные значения модели для каждого из входных примеров (пример - пользовательский). (Inference).
        /// </summary>
        /// <typeparam name="T">Тип данных. Поддерживается <seealso cref="float"/>, <seealso cref="double"/></typeparam>
        /// <param name="source"></param>
        /// <param name="data">Набор примеров для которых вычисляется выход</param>
        /// <param name="device">Устройство для расчетов</param>
        /// <returns></returns>
        public static IEnumerable<T[]> Predict<T>(this Function source,
           IEnumerable<Value> data,
           DeviceDescriptor device) where T : IConvertible
        {
            var inputVariable = source.Inputs.Single(p => p.Name.ToUpper() == "INPUT");
            foreach (var features in data)
            {
                var inputDataMap = new Dictionary<Variable, Value>() { { inputVariable, features } };
                var outputDataMap = new Dictionary<Variable, Value>() { { source.Output, null } };

                source.Evaluate(inputDataMap, outputDataMap, device);

                var predicted = outputDataMap[source.Output].GetDenseData<T>(source.Output);

                foreach (var item in predicted)
                {
                    yield return (T[])item;
                }
            }
        }
        /// <summary>
        /// Вычисляет выходные значения модели для каждого из входных примеров. (Inference).
        /// </summary>
        /// <typeparam name="T">Тип данных. Поддерживается <seealso cref="float"/>, <seealso cref="double"/></typeparam>
        /// <param name="source"></param>
        /// <param name="data">Набор примеров для которых вычисляется выход</param>
        /// <param name="device">Устройство для расчетов</param>
        /// <returns></returns>
        public static IEnumerable<T[]> Predict<T>(this Function source,
            IEnumerable<T[]> data,
            DeviceDescriptor device,
            int minibatchSize = 512) where T : IConvertible
        {
            DataConverter valueConverter = new DataConverter(device);
            var values = valueConverter.ConvertDataToValue(data, minibatchSize);
            return source.Predict<T>(values, device);
        }
        /// <summary>
        /// Вычисляет выходные значения модели для каждого из входных примеров (пример - последовательность). (Inference).
        /// </summary>
        /// <typeparam name="T">Тип данных. Поддерживается <seealso cref="float"/>, <seealso cref="double"/></typeparam>
        /// <param name="source"></param>
        /// <param name="device">Устройство для расчетов</param>
        /// <returns></returns>
        public static IEnumerable<T[]> Predict<T>(this Function source,
            IEnumerable<IList<T[]>> data,
            DeviceDescriptor device,
            int minibatchSize = 512) where T : IConvertible
        {
            DataConverter valueConverter = new DataConverter(device);
            var values = valueConverter.ConvertDataToValue(data, minibatchSize);
            return source.Predict<T>(values, device);
        }
        /// <summary>
        /// Вычисляет выходные значения модели для каждого из входных примеров (пример - 2D). (Inference).
        /// </summary>
        /// <typeparam name="T">Тип данных. Поддерживается <seealso cref="float"/>, <seealso cref="double"/></typeparam>
        /// <param name="source"></param>
        /// <param name="device">Устройство для расчетов</param>
        /// <returns></returns>
        public static IEnumerable<T[]> Predict<T>(this Function source,
            IEnumerable<T[,]> data,
            DeviceDescriptor device,
            int minibatchSize = 512) where T : IConvertible
        {
            DataConverter valueConverter = new DataConverter(device);
            var values = valueConverter.ConvertDataToValue(data, minibatchSize);
            return source.Predict<T>(values, device);
        }

        /// <summary>
        /// Вычисляет выходные значения модели одного примера (пример - пользовательский). (Inference).
        /// </summary>
        /// <typeparam name="T">Тип данных. Поддерживается <seealso cref="float"/>, <seealso cref="double"/></typeparam>
        /// <param name="source"></param>
        /// <param name="device">Устройство для расчетов</param>
        /// <returns></returns>
        public static T[] Predict<T>(this Function source, Value data, DeviceDescriptor device) where T:IConvertible
        {
            return source.Predict<T>(Enumerable.Repeat(data, 1), device).FirstOrDefault();
        }
        /// <summary>
        /// Вычисляет выходные значения модели одного примера (пример - последовательность). (Inference).
        /// </summary>
        /// <typeparam name="T">Тип данных. Поддерживается <seealso cref="float"/>, <seealso cref="double"/></typeparam>
        /// <param name="source"></param>
        /// <param name="device">Устройство для расчетов</param>
        /// <returns></returns>
        public static T[] Predict<T>(this Function source, T[] data, DeviceDescriptor device) where T : IConvertible
        {
            return source.Predict<T>(Enumerable.Repeat(data, 1), device).FirstOrDefault();
        }
        /// <summary>
        /// Вычисляет выходные значения модели одного примера. (Inference).
        /// </summary>
        /// <typeparam name="T">Тип данных. Поддерживается <seealso cref="float"/>, <seealso cref="double"/></typeparam>
        /// <param name="source"></param>
        /// <param name="device">Устройство для расчетов</param>
        /// <returns></returns>
        public static T[] Predict<T>(this Function source, IList<T[]> data, DeviceDescriptor device) where T : IConvertible
        {
            return source.Predict<T>(Enumerable.Repeat(data, 1), device).FirstOrDefault();
        }
        /// <summary>
        /// Вычисляет выходные значения модели одного примера (пример - 2D). (Inference).
        /// </summary>
        /// <typeparam name="T">Тип данных. Поддерживается <seealso cref="float"/>, <seealso cref="double"/></typeparam>
        /// <param name="source"></param>
        /// <param name="device">Устройство для расчетов</param>
        /// <returns></returns>
        public static T[] Predict<T>(this Function source, T[,] data, DeviceDescriptor device) where T : IConvertible
        {
            return source.Predict<T>(Enumerable.Repeat(data, 1), device).FirstOrDefault();
        }
        #endregion

        #endregion

        #region Sequential extensions   

        #region Evaluate<T>
        /// <summary>
        /// Вычисляет выход модели для каждого из тестовых примеров.
        /// </summary>
        /// <typeparam name="T">Тип данных. Поддерживается <seealso cref="float"/>, <seealso cref="double"/></typeparam>
        /// <param name="source"></param>
        /// <param name="testData">Тестовые данные. Каждый минипакет должен содержать 1 тестовый пример.</param>
        /// <param name="device">Устройство для расчетов</param>
        /// <param name="minibatchSize">Размер минипакета для оценки. Использование позволяет оценивать данные пачками(параллельно), не тратя ресурсы на пересылку данных в память. Оптимальный размер зависит от объема данных, доступной памяти GPU (лучшее ускорение).</param>
        /// <returns></returns>
        public static IEnumerable<EvaluateItem<T>> Evaluate<T>(this Sequential<T> source,
            IEnumerable<Minibatch> testData,
            DeviceDescriptor device) where T : IConvertible
        {
            return source.Model.Evaluate<T>(testData, device);
        }
        /// <summary>
        /// Вычисляет выход модели для каждого из тестовых примеров.
        /// </summary>
        /// <typeparam name="T">Тип данных. Поддерживается <seealso cref="float"/>, <seealso cref="double"/></typeparam>
        /// <param name="source"></param>
        /// <param name="testData">Набор тестовых данных для обучения. Каждый пример должен содержать в начале массива признаки размерностью inputDim, а в конце метки классов размерностью outputDim.
        /// Например inputDim = 3, outputDim = 2: [f1, f2, f3, l1, l2]</param>
        /// <param name="inputDim">Размерность признаков (разрядность)</param>
        /// <param name="device">Устройство для расчетов</param>
        /// <param name="minibatchSize">Размер минипакета для оценки. Использование позволяет оценивать данные пачками(параллельно), не тратя ресурсы на пересылку данных в память. Оптимальный размер зависит от объема данных, доступной памяти GPU (лучшее ускорение).</param>
        /// <returns></returns>
        public static IEnumerable<EvaluateItem<T>> Evaluate<T>(this Sequential<T> source,
            IEnumerable<T[]> testData,
            int inputDim,
            DeviceDescriptor device,
            int minibatchSize = 512) where T : IConvertible
        {
            return source.Model.Evaluate<T>(testData, inputDim, device, minibatchSize);
        }
        /// <summary>
        /// Вычисляет выход модели для каждого из тестовых примеров.
        /// </summary>
        /// <typeparam name="T">Тип данных. Поддерживается <seealso cref="float"/>, <seealso cref="double"/></typeparam>
        /// <param name="source"></param>
        /// <param name="features">Набор тестовых последовательностей (признаков). Каждая последовательность может быть переменной длинны, но одинаковой размерности (массивы из которых состоит последовательность, должны иметь одинаковую длину)</param>
        /// <param name="labels">Набор тестовых меток. Размерность меток должна быть одинаковая.</param>
        /// <param name="device">Устройство для расчетов</param>
        /// <param name="minibatchSize">Размер минипакета для оценки. Использование позволяет оценивать данные пачками(параллельно), не тратя ресурсы на пересылку данных в память. Оптимальный размер зависит от объема данных, доступной памяти GPU (лучшее ускорение).</param>
        /// <returns></returns>
        public static IEnumerable<EvaluateItem<T>> Evaluate<T>(this Sequential<T> source,
            IEnumerable<IList<T[]>> features,
            IEnumerable<T[]> labels,
            DeviceDescriptor device,
            int minibatchSize = 512) where T : IConvertible
        {
            return source.Model.Evaluate<T>(features, labels, device, minibatchSize);
        }
        /// <summary>
        /// Вычисляет выход модели для каждого из тестовых примеров.
        /// </summary>
        /// <typeparam name="T">Тип данных. Поддерживается <seealso cref="float"/>, <seealso cref="double"/></typeparam>
        /// <param name="source"></param>
        /// <param name="features">Набор тестовых данных</param>
        /// <param name="device">Устройство для расчетов</param>
        /// <param name="minibatchSize">Размер минипакета для оценки. Использование позволяет оценивать данные пачками(параллельно), не тратя ресурсы на пересылку данных в память. Оптимальный размер зависит от объема данных, доступной памяти GPU (лучшее ускорение).</param>
        /// <returns></returns>
        public static IEnumerable<EvaluateItem<T>> Evaluate<T>(this Sequential<T> source,
            IEnumerable<T[,]> features,
            IEnumerable<T[]> labels,
            DeviceDescriptor device,
            int minibatchSize = 512) where T : IConvertible
        {
            return source.Model.Evaluate<T>(features, labels, device, minibatchSize);
        }
        #endregion

        #region Predict<T>

        /// <summary>
        /// Вычисляет выходные значения модели одного примера (пример - пользовательский). (Inference).
        /// </summary>
        /// <typeparam name="T">Тип данных. Поддерживается <seealso cref="float"/>, <seealso cref="double"/></typeparam>
        /// <param name="source"></param>
        /// <param name="device">Устройство для расчетов</param>
        /// <returns></returns>
        public static T[] Predict<T>(this Sequential<T> source, Value data, DeviceDescriptor device) where T : IConvertible
        {
            return source.Model.Predict<T>(data, device);
        }
        /// <summary>
        /// Вычисляет выходные значения модели для каждого из входных примеров. (Inference).
        /// </summary>
        /// <typeparam name="T">Тип данных. Поддерживается <seealso cref="float"/>, <seealso cref="double"/></typeparam>
        /// <param name="source"></param>
        /// <param name="data">Набор примеров для которых вычисляется выход</param>
        /// <param name="device">Устройство для расчетов</param>
        /// <returns></returns>
        public static IEnumerable<T[]> Predict<T>(this Sequential<T> source,
           IEnumerable<Value> data,
           DeviceDescriptor device) where T : IConvertible
        {
            return source.Model.Predict<T>(data, device);
        }
        /// <summary>
        /// Вычисляет выходные значения модели для каждого из входных примеров. (Inference).
        /// </summary>
        /// <typeparam name="T">Тип данных. Поддерживается <seealso cref="float"/>, <seealso cref="double"/></typeparam>
        /// <param name="source"></param>
        /// <param name="data">Набор примеров для которых вычисляется выход</param>
        /// <param name="device
        public static IEnumerable<T[]> Predict<T>(this Sequential<T> source,
            IEnumerable<T[]> data,
            DeviceDescriptor device,
            int minibatchSize = 512) where T : IConvertible
        {           
            return source.Model.Predict<T>(data, device, minibatchSize);
        }
        /// <summary>
        /// Вычисляет выходные значения модели для каждого из входных примеров (пример - последовательность). (Inference).
        /// </summary>
        /// <typeparam name="T">Тип данных. Поддерживается <seealso cref="float"/>, <seealso cref="double"/></typeparam>
        /// <param name="source"></param>
        /// <param name="device">Устройство для расчетов</param>
        /// <returns></returns>
        public static IEnumerable<T[]> Predict<T>(this Sequential<T> source,
            IEnumerable<IList<T[]>> data,
            DeviceDescriptor device,
            int minibatchSize = 512) where T : IConvertible
        {            
            return source.Model.Predict<T>(data, device, minibatchSize);
        }
        /// <summary>
        /// Вычисляет выходные значения модели для каждого из входных примеров (пример - 2D). (Inference).
        /// </summary>
        /// <typeparam name="T">Тип данных. Поддерживается <seealso cref="float"/>, <seealso cref="double"/></typeparam>
        /// <param name="source"></param>
        /// <param name="device">Устройство для расчетов</param>
        /// <returns></returns>
        public static IEnumerable<T[]> Predict<T>(this Sequential<T> source,
            IEnumerable<T[,]> data,
            DeviceDescriptor device,
            int minibatchSize = 512) where T : IConvertible
        {           
            return source.Model.Predict<T>(data, device, minibatchSize);
        }
        /// <summary>
        /// Вычисляет выходные значения модели одного примера. (Inference).
        /// </summary>
        /// <typeparam name="T">Тип данных. Поддерживается <seealso cref="float"/>, <seealso cref="double"/></typeparam>
        /// <param name="source"></param>
        /// <param name="device">Устройство для расчетов</param>
        /// <returns></returns>
        public static T[] Predict<T>(this Sequential<T> source, T[] data, DeviceDescriptor device) where T : IConvertible
        {
            return source.Model.Predict<T>(data, device);
        }
        /// <summary>
        /// Вычисляет выходные значения модели одного примера (пример - последовательность). (Inference).
        /// </summary>
        /// <typeparam name="T">Тип данных. Поддерживается <seealso cref="float"/>, <seealso cref="double"/></typeparam>
        /// <param name="source"></param>
        /// <param name="device">Устройство для расчетов</param>
        /// <returns></returns>
        public static T[] Predict<T>(this Sequential<T> source, IList<T[]> data, DeviceDescriptor device) where T : IConvertible
        {
            return source.Model.Predict<T>(data, device);
        }
        /// <summary>
        /// Вычисляет выходные значения модели одного примера (пример - 2D). (Inference).
        /// </summary>
        /// <typeparam name="T">Тип данных. Поддерживается <seealso cref="float"/>, <seealso cref="double"/></typeparam>
        /// <param name="source"></param>
        /// <param name="device">Устройство для расчетов</param>
        /// <returns></returns>
        public static T[] Predict<T>(this Sequential<T> source, T[,] data, DeviceDescriptor device) where T : IConvertible
        {
            return source.Model.Predict<T>(data, device);
        }
        #endregion

        #endregion
    }
}

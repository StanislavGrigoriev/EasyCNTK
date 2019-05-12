using CNTK;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using EasyCNTK.Learning.Metrics;

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
        public static IList<RegressionMetrics> GetRegressionMetrics<T>(this IEnumerable<EvaluateItem<T>> source)
        {
            var firstItem = source.FirstOrDefault();
            if (firstItem.Equals(default(EvaluateItem<T>)))
            {
                return new List<RegressionMetrics>();
            }
            if (firstItem.EvaluatedValue.Count != firstItem.ExpectedValue.Count) throw new ArgumentException($"Несоответсвие размерности ожидаемых({firstItem.ExpectedValue.Count}) и оцененных({firstItem.EvaluatedValue.Count}) значений.");

            var result = firstItem.EvaluatedValue
                .Select(p => new RegressionMetrics())
                .ToArray();
            var evaluateDataAcummulator = new T[result.Length];
            var expectedDataAccumulator = new T[result.Length];
            int countItems = 0;
            foreach (var item in source)
            {
                for (int i = 0; i < result.Length; i++)
                {
                    dynamic evaluated = item.EvaluatedValue[i];
                    dynamic expected = item.ExpectedValue[i];
                    var mae = Math.Abs(evaluated - expected);
                    var rmse = Math.Pow(evaluated - expected, 2);
                    checked
                    {
                        result[i].MAE += mae;
                        result[i].RMSE += rmse;
                        evaluateDataAcummulator[i] += evaluated;
                        expectedDataAccumulator[i] += expected;
                    }
                }

                countItems++;
            }
            for (int i = 0; i < result.Length; i++)
            {
                evaluateDataAcummulator[i] = (dynamic)evaluateDataAcummulator[i] / countItems;
                expectedDataAccumulator[i] = (dynamic)expectedDataAccumulator[i] / countItems;
            }

            var evaluateVarianceAcummulator = new T[result.Length];
            var expectedVarianceAccumulator = new T[result.Length];
            foreach (var item in source)
            {
                for (int i = 0; i < result.Length; i++)
                {
                    dynamic evaluated = item.EvaluatedValue[i];
                    dynamic expected = item.ExpectedValue[i];
                    checked
                    {
                        evaluateVarianceAcummulator[i] += Math.Pow(evaluated - expected, 2);
                        expectedVarianceAccumulator[i] += Math.Pow(expected - expectedDataAccumulator[i], 2);
                    }
                }
            }

            for (int i = 0; i < result.Length; i++)
            {
                result[i].MAE = result[i].MAE / countItems;
                result[i].RMSE = Math.Sqrt(result[i].RMSE / countItems);
                result[i].Determination = 1 - ((dynamic)evaluateVarianceAcummulator[i] / expectedVarianceAccumulator[i]);
            }

            return result;
        }
        /// <summary>
        /// Вычисляет метрики для задач бинарной классификации. 
        /// Подразумевается, что выход закодирован в One-Hot-Encoding(и обернут в Softmax, хотя возможно использовать <seealso cref="ActivationFunctions.Sigmoid"/>, <seealso cref="ActivationFunctions.HardSigmoid"/>), в ином случае метрика рассчитается некорректно.
        /// </summary>
        /// <typeparam name="T">Тип данных. Поддерживается <seealso cref="float"/>, <seealso cref="double"/></typeparam>
        /// <param name="source"></param>
        /// <returns></returns>
        public static BinaryClassificationMetrics GetBinaryClassificationMetrics<T>(this IEnumerable<EvaluateItem<T>> source)
        {
            int TP = 0; //факт 1, оценка 1
            int FP = 0; //факт 0, оценка 0
            int FN = 0; //факт 1, оценка 0
            int TN = 0; //факт 0, оценка 1

            foreach (var item in source)
            {
                bool isPositive = (dynamic)item.ExpectedValue[0] == 0;
                var expected = item.ExpectedValue.IndexOf(item.ExpectedValue.Max());
                var evaluated = item.EvaluatedValue.IndexOf(item.EvaluatedValue.Max());

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
                        FP++;
                    }
                    else
                    {
                        TN++;
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
        /// Вычисляет метрики для задач многоклассовой классификации. 
        /// Подразумевается, что выход закодирован в One-Hot-Encoding(и обернут в Softmax, хотя возможно использовать <seealso cref="ActivationFunctions.Sigmoid"/>, <seealso cref="ActivationFunctions.HardSigmoid"/>), в ином случае метрика рассчитается некорректно.
        /// </summary>
        /// <typeparam name="T">Тип данных. Поддерживается <seealso cref="float"/>, <seealso cref="double"/></typeparam>
        /// <param name="source"></param>
        /// <returns></returns>
        public static ClassificationMetrics GetClassificationMetrics<T>(this IEnumerable<EvaluateItem<T>> source) 
        {
            var firstElement = source.FirstOrDefault();
            if (firstElement.Equals(default(EvaluateItem<T>)))
            {
                throw new ArgumentException("Последовательность IEnumerable<EvaluateItem<T>> не содержит элементов.", "source");
            }
            
            var confusionMatrix = new double[firstElement.EvaluatedValue.Count, firstElement.EvaluatedValue.Count];
            int countAccurateSamples = 0;
            int countSamples = 0;
            foreach (var item in source)
            {
                var expected = item.ExpectedValue.IndexOf(item.ExpectedValue.Max());
                var evaluated = item.EvaluatedValue.IndexOf(item.EvaluatedValue.Max());

                confusionMatrix[expected, evaluated]++;
                if (expected == evaluated) countAccurateSamples++;
                
                countSamples++;
            }
            for (int i = 0; i < firstElement.EvaluatedValue.Count; i++)
                for (int j = 0; j < firstElement.EvaluatedValue.Count; j++)
                {
                    confusionMatrix[i, j] /= countSamples;
                }
            double accuracy = (double)countAccurateSamples / countSamples;
            return new ClassificationMetrics()
            {
                Accuracy = accuracy,
                ConfusionMatrix = confusionMatrix
            };
        }


        #region Function extensions

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
            DeviceDescriptor device)
        {
            var inputVariable = source.Inputs.Single(p => p.Name.ToUpper() == "INPUT");            
            foreach (var miniBatch in testData)
            {
                var inputDataMap = new Dictionary<Variable, Value>() { { inputVariable, miniBatch.Features } };
                var outputDataMap = new Dictionary<Variable, Value>() { { source.Output, null } };

                source.Evaluate(inputDataMap, outputDataMap, device);

                var expected = miniBatch.Labels.GetDenseData<T>(source.Output);
                var evaluated = outputDataMap[source.Output].GetDenseData<T>(source.Output);

                var evaluateItem = new EvaluateItem<T>(expected[0], evaluated[0]);
                yield return evaluateItem;
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
        /// <returns></returns>
        public static IEnumerable<EvaluateItem<T>> Evaluate<T>(this Function source,
            IEnumerable<IList<T>> testData,
            int inputDim,
            DeviceDescriptor device)
        {
            ValueConverter valueConverter = new ValueConverter();
            var test = valueConverter.ConvertDatasetToMinibatch(testData, inputDim, 1, device);
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
        /// <returns></returns>
        public static IEnumerable<EvaluateItem<T>> Evaluate<T>(this Function source,
            IEnumerable<IList<T[]>> features,
            IEnumerable<T[]> labels,
            DeviceDescriptor device)
        {
            ValueConverter valueConverter = new ValueConverter();
            var test = valueConverter.ConvertDatasetToMinibatch(features, labels, 1, device);
            return source.Evaluate<T>(test, device);
        }
        /// <summary>
        /// Вычисляет выход модели для каждого из тестовых примеров.
        /// </summary>
        /// <typeparam name="T">Тип данных. Поддерживается <seealso cref="float"/>, <seealso cref="double"/></typeparam>
        /// <param name="source"></param>
        /// <param name="testData">Набор тестовых данных</param>
        /// <param name="device">Устройство для расчетов</param>
        /// <returns></returns>
        public static IEnumerable<EvaluateItem<T>> Evaluate<T>(this Function source,
            IEnumerable<Sample2D<T>> testData,
            DeviceDescriptor device)
        {
            ValueConverter valueConverter = new ValueConverter();
            var test = valueConverter.ConvertDatasetToMinibatch(testData, 1, device);
            return source.Evaluate<T>(test, device);
        }
        #endregion

        #region Sequential extensions   
        /// <summary>
        /// Вычисляет выход модели для каждого из тестовых примеров.
        /// </summary>
        /// <typeparam name="T">Тип данных. Поддерживается <seealso cref="float"/>, <seealso cref="double"/></typeparam>
        /// <param name="source"></param>
        /// <param name="testData">Тестовые данные. Каждый минипакет должен содержать 1 тестовый пример.</param>
        /// <param name="device">Устройство для расчетов</param>
        /// <returns></returns>
        public static IEnumerable<EvaluateItem<T>> Evaluate<T>(this Sequential<T> source,
            IEnumerable<Minibatch> testData,
            DeviceDescriptor device)
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
        /// <returns></returns>
        public static IEnumerable<EvaluateItem<T>> Evaluate<T>(this Sequential<T> source,
            IEnumerable<IList<T>> testData,
            int inputDim,
            DeviceDescriptor device)
        {
            return source.Model.Evaluate<T>(testData, inputDim, device);
        }
        /// <summary>
        /// Вычисляет выход модели для каждого из тестовых примеров.
        /// </summary>
        /// <typeparam name="T">Тип данных. Поддерживается <seealso cref="float"/>, <seealso cref="double"/></typeparam>
        /// <param name="source"></param>
        /// <param name="features">Набор тестовых последовательностей (признаков). Каждая последовательность может быть переменной длинны, но одинаковой размерности (массивы из которых состоит последовательность, должны иметь одинаковую длину)</param>
        /// <param name="labels">Набор тестовых меток. Размерность меток должна быть одинаковая.</param>
        /// <param name="device">Устройство для расчетов</param>
        /// <returns></returns>
        public static IEnumerable<EvaluateItem<T>> Evaluate<T>(this Sequential<T> source,
            IEnumerable<IList<T[]>> features,
            IEnumerable<T[]> labels,
            DeviceDescriptor device)
        {
            return source.Model.Evaluate<T>(features, labels, device);
        }
        /// <summary>
        /// Вычисляет выход модели для каждого из тестовых примеров.
        /// </summary>
        /// <typeparam name="T">Тип данных. Поддерживается <seealso cref="float"/>, <seealso cref="double"/></typeparam>
        /// <param name="source"></param>
        /// <param name="testData">Набор тестовых данных</param>
        /// <param name="device">Устройство для расчетов</param>
        /// <returns></returns>
        public static IEnumerable<EvaluateItem<T>> Evaluate<T>(this Sequential<T> source,
            IEnumerable<Sample2D<T>> testData,
            DeviceDescriptor device)
        {
            return source.Model.Evaluate<T>(testData, device);
        } 
        #endregion
    }
}

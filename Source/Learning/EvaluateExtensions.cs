using CNTK;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using EasyCNTK.Learning.Metrics;
using System.Globalization;

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
                        expectedDataAccumulator[i] += expected;
                    }
                }

                countItems++;
            }
            for (int i = 0; i < result.Length; i++)
            {
                expectedDataAccumulator[i] = (dynamic)expectedDataAccumulator[i] / countItems;
            }

            var expectedVarianceAccumulator = new T[result.Length];
            foreach (var item in source)
            {
                for (int i = 0; i < result.Length; i++)
                {
                    dynamic expected = item.ExpectedValue[i];
                    checked
                    {
                        expectedVarianceAccumulator[i] += Math.Pow(expected - expectedDataAccumulator[i], 2);
                    }
                }
            }

            for (int i = 0; i < result.Length; i++)
            {
                result[i].Determination = 1 - (result[i].RMSE / (dynamic)expectedVarianceAccumulator[i]);
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
        public static BinaryClassificationMetrics GetBinaryClassificationMetrics<T>(this IEnumerable<EvaluateItem<T>> source, double threshold = 0.5)
        {
            int TP = 0; //факт 1, оценка 1
            int TN = 0; //факт 0, оценка 0
            int FP = 0; //факт 0, оценка 1
            int FN = 0; //факт 1, оценка 0

            foreach (var item in source)
            {                
                var expected = (int)(dynamic)item.ExpectedValue[0];
                var evaluated = (dynamic)item.EvaluatedValue[0] < threshold ? 0 : 1;

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
                var expected = item.ExpectedValue.IndexOf(item.ExpectedValue.Max());
                var evaluated = item.EvaluatedValue.IndexOf(item.EvaluatedValue.Max());

                classesDistribution[expected].Fraction++;
                confusionMatrix[expected, evaluated]++;
                if (expected == evaluated)
                {
                    countAccurateSamples++;
                    classesDistribution[evaluated].Precision++;
                }
                countSamples++;
            }
            for (int i = 0; i < firstElement.EvaluatedValue.Count; i++)
            {
                classesDistribution[i].Precision /= classesDistribution[i].Fraction;
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
                        classesDistribution[target].Precision++;
                    }
                    countLabels++;
                }                 
            }
            classesDistribution.ForEach(p =>
            {
                p.Precision /= p.Fraction;
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
            IEnumerable<T[]> testData,
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
            IEnumerable<T[]> testData,
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

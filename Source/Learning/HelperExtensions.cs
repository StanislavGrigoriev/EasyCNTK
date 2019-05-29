using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace EasyCNTK.Learning
{
    public static class HelperExtensions
    {
        /// <summary>
        /// Перемешивает данные в коллекции.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="source"></param>
        /// <param name="seed">Начальное значение для генератора случайных чисел (<seealso cref="Random"/>), если 0 - используется генератор по умолчанию </param>
        public static void Shuffle<T>(this IList<T> source, int seed = 0)
        {
            Random random = new Random(seed);
            int countLeft = source.Count;
            while (countLeft > 1)
            {
                countLeft--;
                int indexNextItem = random.Next(countLeft + 1);
                T temp = source[indexNextItem];
                source[indexNextItem] = source[countLeft];
                source[countLeft] = temp;
            }
        }
        /// <summary>
        /// Разбивает набор данных на 2 части в заданном соотношении
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="source">Исходный набор данных</param>
        /// <param name="percent">Размер первого набора данных в процентах от исходной коллекции. Должен быть в диапазоне [0;1].</param>
        /// <param name="first">Первый набор данных</param>
        /// <param name="second">Второй набор данных</param>
        /// <param name="randomizeSplit">Случайное разбиение (данные для наборов берутся случайно из всей выборки)</param>        
        /// <param name="seed">Начальное значение для генератора случайных чисел, если 0 - используется генератор по умолчанию</param>
        public static void Split<T>(this IList<T> source, double percent, out IList<T> first, out IList<T> second, bool randomizeSplit = false, int seed = 0)
        {
            if (percent > 1 || percent < 0)
            {
                throw new ArgumentOutOfRangeException("Percent must be in range [0;1]", "percent");
            }
            int firstCount = (int)(source.Count * percent);

            if (randomizeSplit)
            {
                source.Shuffle(seed);
            }
            first = new List<T>(source.Take(firstCount));
            second = new List<T>(source.Skip(firstCount));
        }
        /// <summary>
        /// Разбивает набор данных на 2 части в заданном соотношении, сохраняя исходное распределение классов неизменным для обоих коллекций. Подразумевает, что один пример содержит один класс (Задача одноклассовой классификации).
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <typeparam name="U"></typeparam>
        /// <param name="source">Исходный набор данных</param>
        /// <param name="percent">Размер первого набора данных в процентах от исходной коллекции. Должен быть в диапазоне [0;1].</param>
        /// <param name="labelSelector">Селектор, извлекающий метку класса</param>
        /// <param name="labelComparer">Компаратор, используется для определения равенства меток двух классов</param>
        /// <param name="first">Первый набор данных</param>
        /// <param name="second">Второй набор данных</param>
        /// <param name="randomizeSplit">>Случайное разбиение (данные для наборов берутся случайно из всей выборки)</param>
        /// <param name="seed">Начальное значение для генератора случайных чисел, если 0 - используется генератор по умолчанию</param>
        public static void SplitBalanced<T, U>(this IList<T> source, double percent, Func<T, U> labelSelector, IEqualityComparer<U> labelComparer, out IList<T> first, out IList<T> second, bool randomizeSplit = false, int seed = 0)
        {
            if (percent > 1 || percent < 0)
            {
                throw new ArgumentOutOfRangeException("Percent must be in range [0;1]", "percent");
            }
            if (source.Count < 2)
            {
                throw new ArgumentOutOfRangeException("Count elements in source collection must be greater 1", "source");
            }

            int firstCount = (int)(source.Count * percent);
            first = new List<T>(firstCount);
            second = new List<T>(source.Count - firstCount);

            if (randomizeSplit)
            {
                source.Shuffle(seed);
            }

            var groupedByLabel = labelComparer == null 
                ? source.GroupBy(labelSelector) 
                : source.GroupBy(labelSelector, labelComparer);
            foreach (var labelGroup in groupedByLabel)
            {
                var labelCount = labelGroup.Count();
                int toFirst = (int)(labelCount * percent);               

                foreach (var item in labelGroup)
                {
                    if (toFirst != 0)
                    {
                        first.Add(item);
                        toFirst--;
                        continue;
                    }
                    second.Add(item);
                }                
            }
        }

        /// <summary>
        /// Разбивает набор данных на 2 части в заданном соотношении, сохраняя исходное распределение классов неизменным для обоих коллекций.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <typeparam name="U"></typeparam>
        /// <param name="source">Исходный набор данных</param>
        /// <param name="percent">Размер первого набора данных в процентах от исходной коллекции. Должен быть в диапазоне [0;1].</param>
        /// <param name="labelSelector">Селектор, извлекающий метку класса</param>
        /// <param name="first">Первый набор данных</param>
        /// <param name="second">Второй набор данных</param>
        /// <param name="randomizeSplit">>Случайное разбиение (данные для наборов берутся случайно из всей выборки)</param>
        /// <param name="seed">Начальное значение для генератора случайных чисел, если 0 - используется генератор по умолчанию</param>
        public static void SplitBalanced<T, U>(this IList<T> source, double percent, Func<T, U> labelSelector, out IList<T> first, out IList<T> second, bool randomizeSplit = false, int seed = 0)
        {
            source.SplitBalanced(percent, labelSelector, null, out first, out second, randomizeSplit, seed);
        }

        /// <summary>
        /// Вычисляет статистику для каждого элемента коллекции. Допускает потерю точности при вычислении значений вне диапазона <seealso cref="double"/>
        /// </summary>
        /// <typeparam name="T">Поддерживается: <seealso cref="int"/>, <seealso cref="long"/>, <seealso cref="float"/>, <seealso cref="double"/>, <seealso cref="decimal"/></typeparam>
        /// <param name="source">Набор данных. Массивы с одинаковой длинной</param>
        /// <param name="epsilon">Разница, на которую должны отличаться 2 числа с плавающей точкой, чтобы считаться разными</param>
        /// <returns></returns>
        public static List<FeatureStatistic> ComputeStatisticForCollection<T>(this IEnumerable<IList<T>> source, double epsilon = 0.5) where T: IConvertible
        {
            var firstElement = source.FirstOrDefault();
            if (firstElement == null)
            {
                return new List<FeatureStatistic>();
            }            
            var result = firstElement
                .Select((p, i) => new FeatureStatistic(epsilon)
                {
                    FeatureName = (i + 1).ToString(),
                    Min = double.MaxValue,
                    Max = double.MinValue
                })
                .ToList();

            int countItems = 0;
            foreach (var item in source)
            {
                countItems++;
                for (int i = 0; i < item.Count; i++)
                {
                    dynamic value = item[i];

                    if (value < result[i].Min)
                    {
                        result[i].Min = value;
                    }
                    if (value > result[i].Max)
                    {
                        result[i].Max = value;
                    }

                    result[i].Average += value;

                    if (result[i].UniqueValues.TryGetValue(value, out int count))
                    {
                        result[i].UniqueValues[value]++;
                    }
                    else
                    {
                        result[i].UniqueValues[value] = 1;
                    }

                }
            }

            result.ForEach(p =>
            {
                p.Average = p.Average / countItems;

                p.MeanAbsoluteDeviation = p.UniqueValues.Aggregate(0.0, (sum, z) => sum + Math.Abs(z.Key - p.Average) * z.Value) / countItems;

                p.Variance = p.UniqueValues.Aggregate(0.0, (sum, z) => sum + Math.Pow(z.Key - p.Average, 2) * z.Value) / countItems;

                p.StandardDeviation = Math.Sqrt(p.Variance);

                #region поиск медианы
                int halfItems = countItems / 2; //половина записей
                int countElements = 0; //накопляемое число элементов при поиске медианы

                for (int i = 0; i < p.UniqueValues.Count; i++)
                {
                    countElements += p.UniqueValues.Values[i];
                    if (countItems % 2 == 0) //четное число элементов
                    {
                        if (countElements == halfItems) // 122|345
                        {
                            p.Median = (p.UniqueValues.Keys[i] + p.UniqueValues.Keys[i + 1]) / 2;
                            break;
                        }
                        else if (countElements > halfItems) //122|225
                        {
                            p.Median = p.UniqueValues.Keys[i];
                            break;
                        }
                    }
                    else
                    {
                        if (countElements >= halfItems + 1) //12|2|34 или 12|2|24  PS +1 потому что halfItems при делении int/int  будет на единицу меньше чем фактическое количество элементов
                        {
                            p.Median = p.UniqueValues.Keys[i];
                            break;
                        }
                    }
                }
                #endregion

            });

            return result;
        }
        /// <summary>
        /// Вычисляет статистику по объекту для каждого свойства типа: <seealso cref="int"/>, <seealso cref="long"/>, <seealso cref="float"/>, <seealso cref="double"/>, <seealso cref="decimal"/>. Допускает потерю точности при вычислении значений вне диапазона <seealso cref="double"/>
        /// </summary>
        /// <typeparam name="TModel"></typeparam>
        /// <param name="source">Набор данных</param>
        /// <param name="withoutProperties">Свойства, для которых не нужно рассчитывать статистику</param>
        /// <param name="epsilon">Разница, на которую должны отличаться 2 числа с плавающей точкой, чтобы считаться разными</param>
        /// <returns></returns>
        public static List<FeatureStatistic> ComputeStatisticForObject<TModel>(this IEnumerable<TModel> source, double epsilon = 0.5, string[] withoutProperties = null) where TModel : class
        {
            if (withoutProperties == null)
            {
                withoutProperties = new string[0];
            }
            var supportedTypes = new Type[] { typeof(int), typeof(long), typeof(float), typeof(double), typeof(decimal) };

            var firstElement = source.FirstOrDefault();
            if (firstElement == null)
            {
                return new List<FeatureStatistic>();
            }

            var properties = firstElement.GetType()
                .GetProperties()
                .Where(p => !withoutProperties.Contains(p.Name))
                .Where(p => supportedTypes.Contains(p.PropertyType))
                .OrderBy(p => p.Name)
                .ToList();

            var result = properties
                .Select(p => new FeatureStatistic(epsilon)
                {
                    FeatureName = p.Name,
                    Min = double.MaxValue,
                    Max = double.MinValue
                })
                .ToList();

            int countItems = 0;
            foreach (var item in source)
            {
                countItems++;
                for (int i = 0; i < properties.Count; i++)
                {
                    var value = Convert.ToDouble(properties[i].GetValue(item));

                    if (value < result[i].Min)
                    {
                        result[i].Min = value;
                    }
                    if (value > result[i].Max)
                    {
                        result[i].Max = value;
                    }

                    result[i].Average += value;

                    if (result[i].UniqueValues.TryGetValue(value, out int count))
                    {
                        result[i].UniqueValues[value]++;
                    }
                    else
                    {
                        result[i].UniqueValues[value] = 1;
                    }

                }
            }

            result.ForEach(p =>
            {
                p.Average = p.Average / countItems;

                p.MeanAbsoluteDeviation = p.UniqueValues.Aggregate(0.0, (sum, z) => sum + Math.Abs(z.Key - p.Average) * z.Value) / countItems;

                p.Variance = p.UniqueValues.Aggregate(0.0, (sum, z) => sum + Math.Pow(z.Key - p.Average, 2) * z.Value) / countItems;

                p.StandardDeviation = Math.Sqrt(p.Variance);

                #region поиск медианы
                int halfItems = countItems / 2; //половина записей
                int countElements = 0; //накопляемое число элементов при поиске медианы

                for (int i = 0; i < p.UniqueValues.Count; i++)
                {
                    countElements += p.UniqueValues.Values[i];
                    if (countItems % 2 == 0) //четное число элементов
                    {
                        if (countElements == halfItems) // 122|345
                        {
                            p.Median = (p.UniqueValues.Keys[i] + p.UniqueValues.Keys[i + 1]) / 2;
                            break;
                        }
                        else if (countElements > halfItems) //122|225
                        {
                            p.Median = p.UniqueValues.Keys[i];
                            break;
                        }
                    }
                    else
                    {
                        if (countElements >= halfItems + 1) //12|2|34 или 12|2|24  PS +1 потому что halfItems при делении int/int  будет на единицу меньше чем фактическое количество элементов
                        {
                            p.Median = p.UniqueValues.Keys[i];
                            break;
                        }
                    }
                }
                #endregion

            });

            return result;
        }
        
        public static IList<double[]> MinMaxNormalize(this IList<double[]> source, bool centerOnXaxis, out double[] mins, out double[] maxes)
        {
            mins = source[0].Select(p => double.MaxValue).ToArray();
            maxes = source[0].Select(p => double.MinValue).ToArray();

            for (int i = 0; i < source.Count; i++)
            {
                for (int j = 0; j < mins.Length; j++)
                {
                    if (source[i][j] > maxes[j])
                    {
                        maxes[j] = source[i][j];
                    }
                    if (source[i][j] < mins[j])
                    {
                        mins[j] = source[i][j];
                    }
                }
            }

            for (int i = 0; i < source.Count; i++)
            {
                for (int j = 0; j < mins.Length; j++)
                {
                    var toZeroPoint = source[i][j] - mins[j];
                    if (centerOnXaxis)
                    {
                        source[i][j] = (toZeroPoint / Math.Abs(maxes[j] - mins[j]) - 0.5) * 2;
                    }
                    else
                    {
                        source[i][j] = toZeroPoint / Math.Abs(maxes[j] - mins[j]);
                    }
                }
            }
            return source;
        }
        /// <summary>
        /// Выполняет нормализацию каждого выбранного свойства по формуле: Xnorm = X / (|Xmax-Xmin|) (приводит Xnorm к диапазону [0;1]). Изменения затрагивают и исходную коллекцию. Поддерживаются типы свойств: <seealso cref="int"/>, <seealso cref="long"/>, <seealso cref="float"/>, <seealso cref="double"/>, <seealso cref="decimal"/>
        /// </summary>
        /// <typeparam name="TModel">Тип класса, для свойств которого требуется произвести нормализацию</typeparam>
        /// <param name="source">Исходная коллекция</param>
        /// <param name="modelMaxes">Экземпляр типа, который инициализрован максимумами соответсующих значений</param>
        /// <param name="modelMins">Экземпляр типа, который инициализрован минимумами соответсующих значений</param>
        /// <param name="propertyNames">Имена свойств, для которых требуется выполнить нормализацию. Если null или пустой, выполняет нормализацию для всех поддерживаемых свойств</param>
        /// <param name="centerOnXaxis">Указывает, требуется ли приводить нормированное значение к диапазону [-1;1]. В случае если значение элемента из исходной коллекции больше/меньше максимума/минимума заданными соответсвующими параметрами, то возможен выход за диапазон [-1;1]</param>
        /// <returns></returns>
        public static IList<TModel> MinMaxNormalize<TModel>(this IList<TModel> source, TModel modelMaxes, TModel modelMins, bool centerOnXaxis, IList<string> propertyNames = null) where TModel : class
        {
            if (propertyNames == null || propertyNames.Count == 0)
            {
                var supportedTypes = new Type[] { typeof(int), typeof(long), typeof(float), typeof(double), typeof(decimal) };
                propertyNames = typeof(TModel)
                    .GetProperties()
                    .Where(p => supportedTypes.Contains(p.PropertyType))
                    .Select(p => p.Name)
                    .ToList();
            }

            var properties = source[0]
                .GetType()
                .GetProperties()
                .Where(p => propertyNames.Contains(p.Name))
                .OrderBy(p => p.Name)
                .ToList();
            var mins = modelMins?
                .GetType()
                .GetProperties()
                .Where(p => propertyNames.Contains(p.Name))
                .OrderBy(p => p.Name)
                .Select(p => Convert.ToDouble(p.GetValue(modelMins)))
                .ToList();
            var maxes = modelMaxes?
                .GetType()
                .GetProperties()
                .Where(p => propertyNames.Contains(p.Name))
                .OrderBy(p => p.Name)
                .Select(p => Convert.ToDouble(p.GetValue(modelMaxes)))
                .ToList();

            if (mins == null || maxes == null)
            {
                mins = properties.Select(p => double.MaxValue).ToList();
                maxes = properties.Select(p => double.MinValue).ToList();

                for (int i = 0; i < source.Count; i++)
                {
                    for (int j = 0; j < mins.Count; j++)
                    {
                        dynamic itemValue = properties[j].GetValue(source[i]);
                        if (itemValue > maxes[j])
                        {
                            maxes[j] = itemValue;
                        }
                        if (itemValue < mins[j])
                        {
                            mins[j] = itemValue;
                        }
                    }
                }
            }

            for (int i = 0; i < source.Count; i++)
            {
                for (int j = 0; j < mins.Count; j++)
                {
                    dynamic itemValue = properties[j].GetValue(source[i]);
                    var toZeroPoint = itemValue - mins[j];
                    var normalValue = toZeroPoint / Math.Abs(maxes[j] - mins[j]);
                    if (centerOnXaxis)
                    {
                        normalValue = (toZeroPoint / Math.Abs(maxes[j] - mins[j]) - 0.5) * 2;
                    }
                    properties[j].SetValue(source[i], normalValue);
                }
            }
            return source;
        }
        /// <summary>
        /// Выполняет нормализацию каждого выбранного свойства по формуле: Xnorm = X / (|Xmax-Xmin|) (приводит Xnorm к диапазону [0;1]). Изменения затрагивают и исходную коллекцию. Поддерживаются типы свойств: <seealso cref="int"/>, <seealso cref="long"/>, <seealso cref="float"/>, <seealso cref="double"/>, <seealso cref="decimal"/> 
        /// </summary>
        /// <typeparam name="TModel"></typeparam>
        /// <param name="source">Исходная коллекция</param>
        /// <param name="centerOnXaxis">Указывает, требуется ли приводить нормированное значение к диапазону [-1;1].</param>
        /// <param name="propertyNames">Имена свойств, для которых требуется выполнить нормализацию. Если null или пустой, выполняет нормализацию для всех поддерживаемых свойств</param>
        /// <returns></returns>
        public static IList<TModel> MinMaxNormalize<TModel>(this IList<TModel> source, bool centerOnXaxis, IList<string> propertyNames = null) where TModel : class
        {
            return source.MinMaxNormalize(null, null, centerOnXaxis, propertyNames);
        }

    }
}

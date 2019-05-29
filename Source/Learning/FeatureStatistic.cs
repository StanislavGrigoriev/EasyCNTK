using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;

namespace EasyCNTK.Learning
{
    /// <summary>
    /// Представляет сбоную статистику по переменной в наборе данных
    /// </summary>
    public class FeatureStatistic
    {
        /// <summary>
        /// Имя свойства класса, представляющего признак/переменную
        /// </summary>
        public string FeatureName { get; set; }    
        /// <summary>
        /// Среднее значение
        /// </summary>
        public double Average { get; set; }
        /// <summary>
        /// Медиана
        /// </summary>
        public double Median { get; set; }
        /// <summary>
        /// Минимальное значение переменной
        /// </summary>
        public double Min { get; set; }
        /// <summary>
        /// Максимальное значение переменной
        /// </summary>
        public double Max { get; set; }
        /// <summary>
        /// Среднеквадратичное отклонение
        /// </summary>
        public double StandardDeviation { get; set; }
        /// <summary>
        /// Дисперсия
        /// </summary>
        public double Variance { get; set; }
        /// <summary>
        /// Среднее абсолютное отклонение
        /// </summary>
        public double MeanAbsoluteDeviation { get; set; }
        /// <summary>
        /// Упорядоченный список уникальных значений переменной. Key - значение переменной, Value - количество переменных с таким значением
        /// </summary>
        public SortedList<double, int> UniqueValues { get; set; } 
        /// <summary>
        /// Инициализирует класс
        /// </summary>
        /// <param name="epsilon">Погрешность. Задает минимум, на который должны отличаться два числа, чтобы считаться разными</param>
        public FeatureStatistic(double epsilon = 0.01)
        {
            UniqueValues = new SortedList<double, int>(new DoubleComparer(epsilon));            
        }

        public override string ToString()
        {
            return $"Name: {FeatureName} Min: {Min:F5} Max: {Max:F5} Average: {Average:F5} MAD: {MeanAbsoluteDeviation:F5} Variance: {Variance:F5} StdDev: {StandardDeviation:F5}";
        }
    }
}
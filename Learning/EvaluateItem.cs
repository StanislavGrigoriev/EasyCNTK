using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace EasyCNTK.Learning
{
    /// <summary>
    /// Представляет реальные и вычисленные моделью значения выхода
    /// </summary>
    /// <typeparam name="T"></typeparam>
    public struct EvaluateItem<T>
    {
        /// <summary>
        /// Реальные значения, ожидаемые на выходе
        /// </summary>
        public IList<T> ExpectedValue { get; set; }
        /// <summary>
        /// Вычисленные моделью значения на выходе
        /// </summary>
        public IList<T> EvaluatedValue { get; set; }
        public EvaluateItem(IList<T> expectedValue, IList<T> evaluatedValue)
        {
            ExpectedValue = expectedValue;
            EvaluatedValue = evaluatedValue;
        }
    }
}

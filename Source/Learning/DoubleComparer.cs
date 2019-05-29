using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace EasyCNTK.Learning
{
    /// <summary>
    /// Выполняет сравнение двух double чисел с заданной погрешностью
    /// </summary>
    public class DoubleComparer : IComparer<double>
    {
        /// <summary>
        /// Погрешность. Задает минимум, на который должны отличаться два числа, чтобы считаться разными
        /// </summary>
        public double Epsilon { get; }
        /// <summary>
        /// Создает экземпляр компаратора
        /// </summary>
        /// <param name="epsilon">Погрешность. Задает минимум, на который должны отличаться два числа, чтобы считаться разными</param>
        public DoubleComparer(double epsilon = 0.01)
        {
            Epsilon = epsilon;
        }

        public int Compare(double x, double y)
        {
            if (Math.Abs(x - y) < Epsilon)
            {
                return 0;
            }
            if (x - y > 0)
            {
                return 1;
            }
            return -1;
        }
    }
}

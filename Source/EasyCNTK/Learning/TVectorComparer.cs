using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Text;

namespace EasyCNTK.Learning
{
    class TVectorComparer<T> : IEqualityComparer<T[]> where T:IConvertible
    {
        DoubleComparer DoubleComparer;

        public TVectorComparer(double epsilon)
        {
            DoubleComparer = new DoubleComparer(epsilon);
        }

        public bool Equals(T[] x, T[] y)
        {
            for (int i = 0; i < x.Length; i++)
            {
                if (DoubleComparer.Compare(x[i].ToDouble(CultureInfo.InvariantCulture), y[i].ToDouble(CultureInfo.InvariantCulture)) != 0)
                    return false;
            }
            return true;
        }

        public int GetHashCode(T[] obj)
        {
            return obj
                .Select((p, i) => p.GetHashCode() ^ i)
                .Aggregate((a, b) => a ^ b);
        }
    }
}

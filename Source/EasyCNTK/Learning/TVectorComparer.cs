using System;
using System.Collections.Generic;
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
            throw new NotImplementedException();
        }

        public int GetHashCode(T[] obj)
        {
            throw new NotImplementedException();
        }
    }
}

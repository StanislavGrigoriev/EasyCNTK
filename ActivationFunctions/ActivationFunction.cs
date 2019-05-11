using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using CNTK;

namespace EasyCNTK.ActivationFunctions
{
    public abstract class ActivationFunction
    {
        public abstract Function ApplyActivationFunction(Function variable, DeviceDescriptor device);
        public abstract string GetDescription();
    }

}

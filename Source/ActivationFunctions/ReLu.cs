using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using CNTK;

namespace EasyCNTK.ActivationFunctions
{
    public class ReLU : ActivationFunction
    {
        public override Function ApplyActivationFunction(Function variable, DeviceDescriptor device)
        {
            return CNTKLib.ReLU(variable);
        }

        public override string GetDescription()
        {
            return "ReLU";
        }
    }
}

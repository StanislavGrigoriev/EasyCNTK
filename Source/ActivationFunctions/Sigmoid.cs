using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using CNTK;

namespace EasyCNTK.ActivationFunctions
{
    public class Sigmoid : ActivationFunction
    {
        public override Function ApplyActivationFunction(Function variable, DeviceDescriptor device)
        {
            return CNTKLib.Sigmoid(variable);
        }

        public override string GetDescription()
        {
            return "Sigmoid";
        }
    }
}

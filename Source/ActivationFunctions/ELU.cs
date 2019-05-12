using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using CNTK;

namespace EasyCNTK.ActivationFunctions
{
    public class ELU : ActivationFunction
    {
        public override Function ApplyActivationFunction(Function variable, DeviceDescriptor device)
        {
            return CNTKLib.ELU(variable);
        }

        public override string GetDescription()
        {
            return "ELU";
        }
    }
}

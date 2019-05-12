using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using CNTK;

namespace EasyCNTK.ActivationFunctions
{
    public class PReLU : ActivationFunction
    {
        public override Function ApplyActivationFunction(Function variable, DeviceDescriptor device)
        {
            var alpha = new Parameter(variable.Output.Shape, variable.Output.DataType, CNTKLib.UniformInitializer(CNTKLib.DefaultParamInitScale), device);
            return CNTKLib.PReLU(alpha, variable);
        }        
        public override string GetDescription()
        {
            return "PReLU";
        }
    }
}

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using CNTK;

namespace EasyCNTK.ActivationFunctions
{
    public class LeakyReLU : ActivationFunction
    {
        private double _alpha;        
        public LeakyReLU(double alpha)
        {
            _alpha = alpha;
        }
        public override Function ApplyActivationFunction(Function variable, DeviceDescriptor device)
        {
            return CNTKLib.LeakyReLU(variable, _alpha);
        }

        public override string GetDescription()
        {
            return $"LeakyReLU(a={_alpha})";
        }
    }
}

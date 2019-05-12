using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using CNTK;

namespace EasyCNTK.ActivationFunctions
{
    public class SELU : ActivationFunction
    {
        private double? _gamma;
        private double? _alpha;
        public SELU() { }
        public SELU(double gamma)
        {
            _gamma = gamma;
        }
        public SELU(double gamma, double alpha)
        {
            _gamma = gamma;
            _alpha = alpha;
        }
        public override Function ApplyActivationFunction(Function variable, DeviceDescriptor device)
        {
            if (_gamma.HasValue && _alpha.HasValue)
            {
                return CNTKLib.SELU(variable, _gamma.Value, _alpha.Value);
            }
            if (_gamma.HasValue)
            {
                return CNTKLib.SELU(variable, _gamma.Value);
            }
            return CNTKLib.SELU(variable);
        }

        public override string GetDescription()
        {
            if (_gamma.HasValue && _alpha.HasValue)
            {
                return $"SELU(g={_gamma}a={_alpha})";
            }
            if (_gamma.HasValue)
            {
                return $"SELU(g={_gamma})";
            }
            return "SELU";
        }
    }
}

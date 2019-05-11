using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using CNTK;

namespace EasyCNTK.ActivationFunctions
{
    public class HardSigmoid : ActivationFunction
    {
        private float _alpha;
        private float _beta;
        public HardSigmoid(float alpha, float beta)
        {
            _alpha = alpha;
            _beta = beta;
        }

        public override Function ApplyActivationFunction(Function variable, DeviceDescriptor device)
        {
            return CNTKLib.HardSigmoid(variable, _alpha, _beta);
        }

        public override string GetDescription()
        {
            return $"HardSigmoid(a={_alpha}b={_beta})";
        }
    }
}

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using CNTK;

namespace EasyCNTK.LossFunctions
{
    public sealed class CrossEntropyWithSoftmaxError : Loss
    {
        private Axis _axis;
        public CrossEntropyWithSoftmaxError()
        {
            _axis = null;
        }
        public CrossEntropyWithSoftmaxError(int axisNumber)
        {
            _axis = new Axis(axisNumber);            
        }
        public override Function GetLoss(Variable prediction, Variable targets)
        {
            if (_axis != null)
            {
                return CNTKLib.CrossEntropyWithSoftmax(prediction, targets, _axis);
            }
            return CNTKLib.CrossEntropyWithSoftmax(prediction, targets);
        }
    }
}

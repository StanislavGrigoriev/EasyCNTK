using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using CNTK;

namespace EasyCNTK.LossFunctions
{
    public sealed class ClassificationError : Loss
    {
        private int _numberAxis;

        public ClassificationError()
        {
            _numberAxis = -1;
        }
        public ClassificationError(int numberAxis)
        {
            _numberAxis = numberAxis;
        }
        public override Function GetLoss(Variable prediction, Variable targets)
        {
            if (_numberAxis == -1)
            {
                return CNTKLib.ClassificationError(prediction, targets);
            }
            return CNTKLib.ClassificationError(prediction, targets, new Axis(_numberAxis));
        }
    }
}

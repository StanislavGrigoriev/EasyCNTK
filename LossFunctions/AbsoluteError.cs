using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using CNTK;

namespace EasyCNTK.LossFunctions
{
    public sealed class AbsoluteError : Loss
    {
        public override Function GetLoss(Variable prediction, Variable targets)
        {
            var absolute = CNTKLib.Minus(prediction, targets);
            return CNTKLib.Abs(absolute);
        }
    }
}

using CNTK;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace EasyCNTK.LossFunctions
{
    public abstract class Loss
    {
        public abstract Function GetLoss(Variable prediction, Variable targets);
    }
}

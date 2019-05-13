﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using CNTK;

namespace EasyCNTK.LossFunctions
{
    public sealed class BiasError : Loss
    {
        public override Function GetLoss(Variable prediction, Variable targets, DeviceDescriptor device)
        {
            return CNTKLib.Minus(prediction, targets);
        }
    }
}
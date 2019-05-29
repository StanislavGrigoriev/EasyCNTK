//
// Copyright (c) Stanislav Grigoriev. All rights reserved.
// grigorievstas9@gmail.com 
// https://github.com/StanislavGrigoriev/EasyCNTK
//
// Copyright (c) Microsoft. All rights reserved.
//
// Licensed under the MIT license. See LICENSE.txt file in the project root for full license information.
//
using CNTK;

namespace EasyCNTK.LossFunctions
{
    public sealed class AbsoluteError : Loss
    {
        public override Function GetLoss(Variable prediction, Variable targets, DeviceDescriptor device)
        {
            var absolute = CNTKLib.Minus(prediction, targets);
            return CNTKLib.Abs(absolute);
        }
    }
}

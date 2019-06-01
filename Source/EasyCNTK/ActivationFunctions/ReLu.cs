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

namespace EasyCNTK.ActivationFunctions
{
    public class ReLU : ActivationFunction
    {
        public override Function ApplyActivationFunction(Function variable, DeviceDescriptor device)
        {
            return CNTKLib.ReLU(variable);
        }

        public override string GetDescription()
        {
            return "ReLU";
        }
    }
}

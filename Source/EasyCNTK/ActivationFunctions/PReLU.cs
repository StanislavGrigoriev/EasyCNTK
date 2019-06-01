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
    public class PReLU : ActivationFunction
    {
        public override Function ApplyActivationFunction(Function variable, DeviceDescriptor device)
        {
            var alpha = new Parameter(variable.Output.Shape, variable.Output.DataType, CNTKLib.UniformInitializer(CNTKLib.DefaultParamInitScale), device);
            return CNTKLib.PReLU(alpha, variable);
        }        
        public override string GetDescription()
        {
            return "PReLU";
        }
    }
}

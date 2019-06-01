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

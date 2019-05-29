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

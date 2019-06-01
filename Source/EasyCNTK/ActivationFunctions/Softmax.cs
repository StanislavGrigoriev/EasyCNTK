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
    public class Softmax : ActivationFunction
    {
        private int _numberAxis = -1;
        public override Function ApplyActivationFunction(Function variable, DeviceDescriptor device)
        {
            if (_numberAxis == -1)
            {
                return CNTKLib.Softmax(variable);
            }
            return CNTKLib.Softmax(variable, new Axis(_numberAxis));
        }
        public Softmax() { }
        public Softmax(int numberAxis)
        {
            _numberAxis = numberAxis;
        }
        public override string GetDescription()
        {
            return _numberAxis == -1 ? "Softmax" : $"Softmax(axis={_numberAxis})";
        }
    }
}

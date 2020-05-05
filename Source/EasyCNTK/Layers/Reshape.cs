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
using System.Linq;

namespace EasyCNTK.Layers
{
    public sealed class Reshape : Layer
    {
        int[] _targetShape;
        private string _name;
        public Reshape(int[] targetShape, string name = "Reshape")
        {
            _targetShape = targetShape;
            _name = name;
        }
        public override Function Create(Function input, DeviceDescriptor device)
        {
            var resahaped = CNTKLib.Reshape(input, _targetShape, _name);

            return resahaped;
        }

        public override string GetDescription()
        {
            var shape = "";
            _targetShape.ToList().ForEach(p =>
            {
                shape += p.ToString() + "x";
            });
            shape = shape.Substring(0, shape.Length - 1);

            return $"Reshape({shape})";
        }
    }
}

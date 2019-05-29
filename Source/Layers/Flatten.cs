//
// Copyright (c) Stanislav Grigoriev. All rights reserved.
// grigorievstas9@gmail.com 
// https://github.com/StanislavGrigoriev/EasyCNTK
//
// Copyright (c) Microsoft. All rights reserved.
//
// Licensed under the MIT license. See LICENSE.txt file in the project root for full license information.
//
using System.Linq;
using CNTK;

namespace EasyCNTK.Layers
{
    /// <summary>
    /// Реализует слой преобразующий вход в "плоское" представление (вектор)
    /// </summary>
    public sealed class Flatten : Layer
    {
        public override Function Create(Function input, DeviceDescriptor device)
        {
            int newDim = input.Output.Shape.Dimensions.Aggregate((d1, d2) => d1 * d2);
            return CNTKLib.Reshape(input, new int[] { newDim });
        }

        public override string GetDescription()
        {
            return "Flatten";
        }
    }
}

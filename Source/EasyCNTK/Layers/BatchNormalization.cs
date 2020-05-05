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

namespace EasyCNTK.Layers
{
    /// <summary>
    /// Реализует слой батч-нормализации
    /// </summary>
    public sealed class BatchNormalization : Layer
    { 
        private static Function createBatchNorm(Function input, DeviceDescriptor device)
        {
            var scale = new Parameter(input.Output.Shape, input.Output.DataType, 1, device);
            var bias = new Parameter(input.Output.Shape, input.Output.DataType, 0, device);
            var runningMean = new Parameter(input.Output.Shape, input.Output.DataType, 0, device);
            var runningInvStd = new Parameter(input.Output.Shape, input.Output.DataType, 0, device);
            var runningCount = new Parameter(new int[] { 1 }, input.Output.DataType, 0, device);
            return CNTKLib.BatchNormalization(input.Output, scale, bias, runningMean, runningInvStd, runningCount, false);
        }
        /// <summary>
        /// Создает слой батч-нормализации
        /// </summary>
        /// <param name="input"></param>
        /// <param name="device"></param>
        /// <returns></returns>
        public static Function Build(Function input, DeviceDescriptor device)
        {
            return createBatchNorm(input, device);
        }
        public override Function Create(Function input, DeviceDescriptor device)
        {
            return createBatchNorm(input, device);
        }

        public override string GetDescription()
        {
            return "BN";
        }
    }
}

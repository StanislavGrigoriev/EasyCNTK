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
        private string _name;
        private bool _spatial;         
        
        private Function createBatchNorm(Function input, DeviceDescriptor device)
        {
            var scale = new Parameter(input.Output.Shape, input.Output.DataType, 1, device);
            var bias = new Parameter(input.Output.Shape, input.Output.DataType, 0, device);
            var runningMean = new Constant(input.Output.Shape, input.Output.DataType, 0, device);
            var runningInvStd = new Constant(input.Output.Shape, input.Output.DataType, 0, device);
            var runningCount = Constant.Scalar(input.Output.DataType, 0, device);
            var bn = CNTKLib.BatchNormalization(input.Output, scale, bias, runningMean, runningInvStd, runningCount, _spatial);
            return CNTKLib.Alias(bn, _name);
        }
        /// <summary>
        /// Создает слой пакетной нормализации
        /// </summary>
        /// <param name="spatial">Указывает, следует ли вычислять среднее/дисперсию для каждого признака независимо, или в случае сверточных сетей - для каждого фильтра(рекомендуется)</param>
        /// <param name="name"></param>
        public BatchNormalization(bool spatial = false, string name = "BN")
        {
            _spatial = spatial;
            _name = name;
        }

        /// <summary>
        /// Создает слой батч-нормализации
        /// </summary>
        /// <param name="input"></param>
        /// <param name="device"></param>
        /// <returns></returns>
        public static Function Build(Function input, DeviceDescriptor device)
        {
            return new BatchNormalization().createBatchNorm(input, device);
        }
        public override Function Create(Function input, DeviceDescriptor device)
        {
            return createBatchNorm(input, device);
        }

        public override string GetDescription()
        {
            return $"BN(S={_spatial})";
        }
    }
}

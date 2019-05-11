using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
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
            var runningMean = new Constant(input.Output.Shape, input.Output.DataType, 0, device);
            var runningInvStd = new Constant(input.Output.Shape, input.Output.DataType, 0, device);
            var runningCount = new Constant(new int[] { 1 }, input.Output.DataType, 0, device);
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

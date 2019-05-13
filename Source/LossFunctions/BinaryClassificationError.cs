using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using CNTK;

namespace EasyCNTK.LossFunctions
{
    public sealed class BinaryClassificationError : Loss
    {
        private double _thresholdValue;
        public BinaryClassificationError(double threshold = 0.5)
        {
            if (threshold <= 0 || threshold >= 1) throw new ArgumentOutOfRangeException("threshold", "Порог должен быть в диапазоне: (0;1)");
            _thresholdValue = threshold;
        }
        public override Function GetLoss(Variable prediction, Variable targets, DeviceDescriptor device)
        {
            var threshold = new Constant(prediction.Shape, prediction.DataType, _thresholdValue, device);

            var predictionLabel = CNTKLib.Less(prediction, threshold);
            var loss = CNTKLib.Equal(predictionLabel, targets);

            return loss;
        }
    }
}

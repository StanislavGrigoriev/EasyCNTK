using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using CNTK;

namespace EasyCNTK.LossFunctions
{
    public static class Errors
    {
        public static Function BiasError(Variable prediction, Variable targets)
        {
            return CNTKLib.Minus(prediction, targets);
        }
        public static Function AbsoluteError(Variable prediction, Variable targets)
        {
            var absolute = CNTKLib.Minus(prediction, targets);
            return CNTKLib.Abs(absolute);
        }
        public static Function SquaredError(Variable prediction, Variable targets)
        {
            return CNTKLib.SquaredError(prediction, targets);
        }
        public static Function CrossEntropyWithSoftmaxError(Variable prediction, Variable targets)
        {
            return CNTKLib.CrossEntropyWithSoftmax(prediction, targets);
        }
        public static Function CrossEntropyWithSoftmaxError(Variable prediction, Variable targets, Axis axis)
        {
            return CNTKLib.CrossEntropyWithSoftmax(prediction, targets, axis);
        }
        public static Function BinaryCrossEntropyError(Variable prediction, Variable targets)
        {
            return CNTKLib.BinaryCrossEntropy(prediction, targets);
        }
        public static Function WeightedBinaryCrossEntropyError(Variable prediction, Variable targets, Variable weights)
        {
            return CNTKLib.WeightedBinaryCrossEntropy(prediction, targets, weights);
        }

        public static Function ClassificationError(Variable prediction, Variable targets)
        {
            return CNTKLib.ClassificationError(prediction, targets);
        }
        public static Function ClassificationError(Variable prediction, Variable targets, Axis axis)
        {
            return CNTKLib.ClassificationError(prediction, targets, axis);
        }
    }
}

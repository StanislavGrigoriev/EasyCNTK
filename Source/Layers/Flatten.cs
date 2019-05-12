using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
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

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using CNTK;

namespace EasyCNTK.Layers
{
    public abstract class Layer
    {
        public abstract Function Create(Function input, DeviceDescriptor device);
        public abstract string GetDescription();
    }
}

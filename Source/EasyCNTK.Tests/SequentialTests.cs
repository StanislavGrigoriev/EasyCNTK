using System;
using Xunit;
using System.Collections.Generic;
using System.Text;
using CNTK;
using EasyCNTK.Layers;
using EasyCNTK.ActivationFunctions;
using System.Threading.Tasks;

namespace EasyCNTK.Tests
{
    public class SequentialTests
    {
        [Fact]        
        public void IDisposableTest()
        {
            var model = new Sequential<double>(DeviceDescriptor.CPUDevice, new[] { 1 });
            model.Add(new Dense(4, new Tanh()));
            model.CreateInputPointForShortcutConnection("l1");
            model.CreateInputPointForShortcutConnection("l2");
            model.Add(new Dense(4, new Tanh()));
            model.CreateOutputPointForShortcutConnection("l1");
            model.Add(new Dense(4, new Tanh()));

            try
            {
                model.Dispose();
            }
            catch (Exception)
            {
                Assert.True(false);
            }
            Assert.ThrowsAsync<AccessViolationException>(() =>
            {
                return new Task(() => model.Add(new Dense(4, new Tanh())));
            });
        }
        [Fact]
        public void Conv1D_1DShape_Test()
        {
            var model = new Sequential<double>(DeviceDescriptor.CPUDevice, new[] { 10, 1 });
            model.Add(new Convolution1D(4));

            bool shapeIsRight = model.Model.Output.Shape.Dimensions.Count == 2 &&
                model.Model.Output.Shape.Dimensions[0] == 7 &&
                model.Model.Output.Shape.Dimensions[1] == 1;

            Assert.True(shapeIsRight);
        }
        [Fact]
        public void Conv1D_2DShape_Test()
        {
            var model = new Sequential<double>(DeviceDescriptor.CPUDevice, new[] { 10, 5, 1 });
            model.Add(new Convolution1D(4));

            bool shapeIsRight = model.Model.Output.Shape.Dimensions.Count == 3 &&
                model.Model.Output.Shape.Dimensions[0] == 7 &&
                model.Model.Output.Shape.Dimensions[1] == 5 &&
                model.Model.Output.Shape.Dimensions[2] == 1;

            Assert.True(shapeIsRight);
        }
    }
}

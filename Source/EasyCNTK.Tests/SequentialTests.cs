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
    }
}

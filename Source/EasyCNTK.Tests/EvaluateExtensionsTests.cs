using System;
using Xunit;
using EasyCNTK.Learning;
using System.Linq;

namespace EasyCNTK.Tests
{
    public class EvaluateExtensionsTests
    {
        [Fact]
        public void GetRegressionMetrics_R2()
        {
            //arrange
            var data = Enumerable.Range(1, 100)
                .Select(p => new EvaluateItem<double>(new[] { (double)p }, new[] { (double)p }))
                .ToList();
            //act
            var metrics = data.GetRegressionMetrics();
            //assert
            double expected = 1;
            Assert.Equal(metrics[0].Determination, expected);
        }
    }
}

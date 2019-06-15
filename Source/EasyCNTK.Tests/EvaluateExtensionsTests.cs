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
        [Fact]
        public void GetOneLabelClassificationMetrics_Precision()
        {
            var data = new EvaluateItem<double>[]
            {
                new EvaluateItem<double>(new[] { 0.0, 1.0 }, new[] { 0.0, 0.8 }), 
                new EvaluateItem<double>(new[] { 1.0, 0.0 }, new[] { 0.4, 0.9 }), 
                new EvaluateItem<double>(new[] { 1.0, 0.0 }, new[] { 0.7, 0.1 }), 
                new EvaluateItem<double>(new[] { 0.0, 1.0 }, new[] { 0.0, 0.8 }), 
            };

            var metrics = data.GetOneLabelClassificationMetrics();

            Assert.Equal(1, metrics.ClassesDistribution[0].Precision, 0); // 1/1=1
            Assert.Equal(0.67, metrics.ClassesDistribution[1].Precision, 2); // 2/3=0.67
        }
        [Fact]
        public void GetOneLabelClassificationMetrics_Recall()
        {
            var data = new EvaluateItem<double>[]
            {
                new EvaluateItem<double>(new[] { 0.0, 1.0 }, new[] { 0.0, 0.8 }),
                new EvaluateItem<double>(new[] { 1.0, 0.0 }, new[] { 0.4, 0.9 }),
                new EvaluateItem<double>(new[] { 1.0, 0.0 }, new[] { 0.7, 0.1 }),
                new EvaluateItem<double>(new[] { 0.0, 1.0 }, new[] { 0.0, 0.8 }),
            };

            var metrics = data.GetOneLabelClassificationMetrics();

            Assert.Equal(0.5, metrics.ClassesDistribution[0].Recall, 1); // 1/2=0.5
            Assert.Equal(1, metrics.ClassesDistribution[1].Recall, 2); // 2/2=1
        }   
        [Fact]
        public void GetMultiLabelClassificationMetrics_Precision()
        {
            var data = new EvaluateItem<double>[]
            {
                new EvaluateItem<double>(new[] { 0.0, 1.0, 1.0 }, new[] { 0.0, 0.8, 0.6 }),
                new EvaluateItem<double>(new[] { 1.0, 0.0, 1.0 }, new[] { 0.4, 0.9, 0.1 }),
                new EvaluateItem<double>(new[] { 1.0, 0.0, 0.0 }, new[] { 0.7, 0.1, 0.7 }),
                new EvaluateItem<double>(new[] { 1.0, 1.0, 1.0 }, new[] { 0.0, 0.8, 0.2 }),
            };

            var metrics = data.GetMultiLabelClassificationMetrics();

            Assert.Equal(1, metrics.ClassesDistribution[0].Precision); // 1/1=1
            Assert.Equal(0.67, metrics.ClassesDistribution[1].Precision, 2); // 2/3=0.67
            Assert.Equal(0.5, metrics.ClassesDistribution[2].Precision, 1); // 1/2=0.5
        }
        [Fact]
        public void GetMultiLabelClassificationMetrics_Recall()
        {
            var data = new EvaluateItem<double>[]
            {
                new EvaluateItem<double>(new[] { 0.0, 1.0, 1.0 }, new[] { 0.0, 0.8, 0.6 }),
                new EvaluateItem<double>(new[] { 1.0, 0.0, 1.0 }, new[] { 0.4, 0.9, 0.1 }),
                new EvaluateItem<double>(new[] { 1.0, 0.0, 0.0 }, new[] { 0.7, 0.1, 0.7 }),
                new EvaluateItem<double>(new[] { 1.0, 1.0, 1.0 }, new[] { 0.0, 0.8, 0.2 }),
            };

            var metrics = data.GetMultiLabelClassificationMetrics();

            Assert.Equal(0.33, metrics.ClassesDistribution[0].Recall, 2); // 1/3=0.33
            Assert.Equal(1, metrics.ClassesDistribution[1].Recall); // 2/2=1
            Assert.Equal(0.33, metrics.ClassesDistribution[2].Recall, 1); // 1/3=0.33
        }
    }
}

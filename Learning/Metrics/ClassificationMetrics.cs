using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace EasyCNTK.Learning.Metrics
{
    /// <summary>
    /// Метрики оценки многокласовой классификации
    /// </summary>
    public class ClassificationMetrics
    {
        public double Accuracy { get; set; }
        public double[,] ConfusionMatrix { get; set; }
    }
}

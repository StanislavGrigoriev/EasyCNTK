using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace EasyCNTK.Learning.Metrics
{
    /// <summary>
    /// Метрики оценки бинарной классификации
    /// </summary>
    public class BinaryClassificationMetrics
    {
        public double Accuracy { get; set; }
        public double Precision { get; set; }
        public double Recall { get; set; }
        public double F1Score { get; set; }
        public double[,] ConfusionMatix { get; set; }
    }
}

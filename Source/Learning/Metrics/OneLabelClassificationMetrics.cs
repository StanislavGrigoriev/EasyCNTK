using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace EasyCNTK.Learning.Metrics
{
    /// <summary>
    /// Метрики оценки одноклассовой классификации
    /// </summary>
    public class OneLabelClassificationMetrics
    {
        public double Accuracy { get; set; }
        public double[,] ConfusionMatrix { get; set; }
        public List<ClassItem> ClassesDistribution { get; set; }
    }
}

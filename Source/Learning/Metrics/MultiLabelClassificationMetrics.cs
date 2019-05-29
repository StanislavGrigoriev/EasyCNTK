using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace EasyCNTK.Learning.Metrics
{
    /// <summary>
    /// Метрики оценки многоклассовой классификации
    /// </summary>
    public class MultiLabelClassificationMetrics
    {
        public double Accuracy { get; set; }        
        public List<ClassItem> ClassesDistribution { get; set; }
    }
}

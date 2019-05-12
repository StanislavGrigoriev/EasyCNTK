using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace EasyCNTK.Learning.Metrics
{
    /// <summary>
    /// Метрики оценки регрессии
    /// </summary>
    public class RegressionMetrics
    {
        public double MAE { get; set; }
        public double RMSE { get; set; }
        public double Determination { get; set; }
    }    
}

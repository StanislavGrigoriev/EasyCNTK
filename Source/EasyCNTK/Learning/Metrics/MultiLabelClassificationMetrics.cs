//
// Copyright (c) Stanislav Grigoriev. All rights reserved.
// grigorievstas9@gmail.com 
// https://github.com/StanislavGrigoriev/EasyCNTK
//
// Licensed under the MIT license. See LICENSE.txt file in the project root for full license information.
//

using System.Collections.Generic;

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

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
    /// Метрики оценки одноклассовой классификации
    /// </summary>
    public class OneLabelClassificationMetrics
    {
        public double Accuracy { get; set; }
        public double[,] ConfusionMatrix { get; set; }
        public List<ClassItem> ClassesDistribution { get; set; }
    }
}

//
// Copyright (c) Stanislav Grigoriev. All rights reserved.
// grigorievstas9@gmail.com 
// https://github.com/StanislavGrigoriev/EasyCNTK
//
// Licensed under the MIT license. See LICENSE.txt file in the project root for full license information.
//

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

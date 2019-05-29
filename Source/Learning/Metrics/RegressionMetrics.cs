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
    /// Метрики оценки регрессии
    /// </summary>
    public class RegressionMetrics
    {
        public double MAE { get; set; }
        public double RMSE { get; set; }
        public double Determination { get; set; }
    }    
}

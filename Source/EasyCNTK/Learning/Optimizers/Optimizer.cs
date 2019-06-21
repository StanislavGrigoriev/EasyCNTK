//
// Copyright (c) Stanislav Grigoriev. All rights reserved.
// grigorievstas9@gmail.com 
// https://github.com/StanislavGrigoriev/EasyCNTK
//
// Copyright (c) Microsoft. All rights reserved.
//
// Licensed under the MIT license. See LICENSE.txt file in the project root for full license information.
//

using CNTK;
using System.Collections.Generic;

namespace EasyCNTK.Learning.Optimizers
{
    /// <summary>
    /// Базовый класс для реализации оптимизаторов
    /// </summary>
    public abstract class Optimizer
    {
        public abstract double LearningRate { get; }
        public abstract Learner GetOptimizer(IList<Parameter> learningParameters);
        public abstract int MinibatchSize { get; set; }
    }
}

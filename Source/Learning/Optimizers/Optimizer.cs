using CNTK;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace EasyCNTK.Learning.Optimizers
{
    /// <summary>
    /// Базовый класс для реализации оптимизаторов
    /// </summary>
    public abstract class Optimizer
    {
        public abstract double LearningRate { get; }
        public abstract Learner GetOptimizer(IList<Parameter> learningParameters);
    }
}

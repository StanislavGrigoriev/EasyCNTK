﻿using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using CNTK;

namespace EasyCNTK.Learning.Optimizers
{
    /// <summary>
    /// Оптмизатор. Реализует стохастический градиентный спуск c моментом (MomentumSGD)
    /// </summary>
    public sealed class MomentumSGD : Optimizer
    {
        private double _l1RegularizationWeight;
        private double _l2RegularizationWeight;
        private double _gradientClippingThresholdPerSample;
        private int _minibatchSize;
        private double _momentum;
        private bool _unitGain;
        public override double LearningRate { get; }
        /// <summary>
        /// Инициализирует оптимизатор MomentumSGD
        /// </summary>
        /// <param name="learningRate">Скорость обучения</param>
        /// <param name="momentum">Момент</param>
        /// <param name="minibatchSize">Размер минипакета, требуется CNTK чтобы масштабировать параметры оптимизатора для более эффективного обучения</param>
        /// <param name="l1RegularizationWeight">Коэффициент L1 нормы, если 0 - регуляризация не применяется</param>
        /// <param name="l2RegularizationWeight">Коэффициент L2 нормы, если 0 - регуляризация не применяется</param>
        /// <param name="gradientClippingThresholdPerSample">Порог отсечения градиента на каждый пример обучения, используется преимущественно для борьбы с взрывным градиентом в глубоких реккурентных сетях.
        /// По умолчанию установлен в <seealso cref="double.PositiveInfinity"/> - отсечение не используется. Для использования установите необходимый порог.</param>
        /// <param name="unitGain">Указывает, что момент используется в режиме усиления</param> 
        public MomentumSGD(double learningRate,
            double momentum,
            int minibatchSize,
            double l1RegularizationWeight = 0,
            double l2RegularizationWeight = 0,
            double gradientClippingThresholdPerSample = double.PositiveInfinity,            
            bool unitGain = true)
        {
            LearningRate = learningRate;
            _momentum = momentum;
            _l1RegularizationWeight = l1RegularizationWeight;
            _l2RegularizationWeight = l2RegularizationWeight;
            _gradientClippingThresholdPerSample = gradientClippingThresholdPerSample;            
            _unitGain = unitGain;
            _minibatchSize = minibatchSize;
        }
        public override Learner GetOptimizer(IList<Parameter> learningParameters)
        {
            var learningOptions = new AdditionalLearningOptions()
            {
                l1RegularizationWeight = _l1RegularizationWeight,
                l2RegularizationWeight = _l2RegularizationWeight,
                gradientClippingWithTruncation = _gradientClippingThresholdPerSample != double.PositiveInfinity,
                gradientClippingThresholdPerSample = _gradientClippingThresholdPerSample
            };
            return CNTKLib.MomentumSGDLearner(new ParameterVector((ICollection)learningParameters),
                new TrainingParameterScheduleDouble(LearningRate, (uint)_minibatchSize),
                new TrainingParameterScheduleDouble(_momentum, (uint)_minibatchSize),
                _unitGain,
                learningOptions);
        }
    }
}
//
// Copyright (c) Stanislav Grigoriev. All rights reserved.
// grigorievstas9@gmail.com 
// https://github.com/StanislavGrigoriev/EasyCNTK
//
// Copyright (c) Microsoft. All rights reserved.
//
// Licensed under the MIT license. See LICENSE.txt file in the project root for full license information.
//

using System.Collections;
using System.Collections.Generic;
using CNTK;

namespace EasyCNTK.Learning.Optimizers
{
    /// <summary>
    /// Оптимизатор FSAdaGrad. Аналог <seealso cref="Adam"/>
    /// </summary>
    public sealed class FSAdaGrad : Optimizer
    {
        private double _l1RegularizationWeight;
        private double _l2RegularizationWeight;
        private double _gradientClippingThresholdPerSample;
        private double _momentum;
        private double _varianceMomentumSchedule;
        private bool _unitGain;
        public override double LearningRate { get; }
        public override int MinibatchSize { get; set; }

        /// <summary>
        /// Инициализирует оптимизатор FSAdaGrad
        /// </summary>
        /// <param name="learningRate">Скорость обучения</param>
        /// <param name="momentum">Момент</param>
        /// <param name="minibatchSize">Размер минипакета, требуется CNTK чтобы масштабировать параметры оптимизатора для более эффективного обучения. Если равен 0, то будет использован размер митибатча при обучении.</param>
        /// <param name="l1RegularizationWeight">Коэффициент L1 нормы, если 0 - регуляризация не применяется</param>
        /// <param name="l2RegularizationWeight">Коэффициент L2 нормы, если 0 - регуляризация не применяется</param>
        /// <param name="gradientClippingThresholdPerSample">Порог отсечения градиента на каждый пример обучения, используется преимущественно для борьбы с взрывным градиентом в глубоких реккурентных сетях.
        /// По умолчанию установлен в <seealso cref="double.PositiveInfinity"/> - отсечение не используется. Для использования установите необходимый порог.</param>       
        /// <param name="varianceMomentumSchedule">"Beta2" параметр в формуле вычисления момента: http://ruder.io/optimizing-gradient-descent/index.html#adam (аналогично для FSAdaGrad)</param>
        /// <param name="unitGain">Указывает, что момент используется в режиме усиления</param>   
        public FSAdaGrad(double learningRate,
            double momentum,
            int minibatchSize = 0,
            double l1RegularizationWeight = 0,
            double l2RegularizationWeight = 0,
            double gradientClippingThresholdPerSample = double.PositiveInfinity,            
            double varianceMomentumSchedule = 0.9999986111120757,
            bool unitGain = true)
        {
            LearningRate = learningRate;
            _momentum = momentum;
            _l1RegularizationWeight = l1RegularizationWeight;
            _l2RegularizationWeight = l2RegularizationWeight;
            _gradientClippingThresholdPerSample = gradientClippingThresholdPerSample;            
            _varianceMomentumSchedule = varianceMomentumSchedule;
            _unitGain = unitGain;
            MinibatchSize = minibatchSize;
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
            return CNTKLib.FSAdaGradLearner(new ParameterVector((ICollection)learningParameters),
                new TrainingParameterScheduleDouble(LearningRate, (uint)MinibatchSize),
                new TrainingParameterScheduleDouble(_momentum, (uint)MinibatchSize),
                _unitGain,
                new TrainingParameterScheduleDouble(_varianceMomentumSchedule, (uint)MinibatchSize),
                learningOptions);
        }
    }
}

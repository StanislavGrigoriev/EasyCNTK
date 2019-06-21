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
    /// Оптимизатор Adamax. Вариация <seealso cref="Adam"/>
    /// </summary>
    public sealed class Adamax : Optimizer
    {
        private double _momentum;
        private double _l1RegularizationWeight;
        private double _l2RegularizationWeight;
        private double _gradientClippingThresholdPerSample;
        private double _epsilon;
        private double _varianceMomentumSchedule;
        private bool _unitGain;    
        public override double LearningRate { get; }
        public override int MinibatchSize { get; set; }

        /// <summary>
        /// Инициализирует оптимизатор Adamax
        /// </summary>
        /// <param name="learningRate">Скорость обучения</param>
        /// <param name="momentum">Момент</param>
        /// <param name="minibatchSize">Размер минипакета, требуется CNTK чтобы масштабировать параметры оптимизатора для более эффективного обучения. Если равен 0, то будет использован размер митибатча при обучении.</param>
        /// <param name="l1RegularizationWeight">Коэффициент L1 нормы, если 0 - регуляризация не применяется</param>
        /// <param name="l2RegularizationWeight">Коэффициент L2 нормы, если 0 - регуляризация не применяется</param>
        /// <param name="gradientClippingThresholdPerSample">Порог отсечения градиента на каждый пример обучения, используется преимущественно для борьбы с взрывным градиентом в глубоких реккурентных сетях.
        /// По умолчанию установлен в <seealso cref="double.PositiveInfinity"/> - отсечение не используется. Для использования установите необходимый порог.</param>
        /// <param name="epsilon">Константа для стабилизации(защита от деления на 0). Параметр "е" - в формуле правила обновления параметров: http://ruder.io/optimizing-gradient-descent/index.html#adam </param>
        /// <param name="varianceMomentumSchedule">"Beta2" параметр в формуле вычисления момента: http://ruder.io/optimizing-gradient-descent/index.html#adam</param>
        /// <param name="unitGain">Указывает, что момент используется в режиме усиления</param>        
        public Adamax(double learningRate,
            double momentum,
            int minibatchSize = 0,
            double l1RegularizationWeight = 0,
            double l2RegularizationWeight = 0,
            double gradientClippingThresholdPerSample = double.PositiveInfinity,
            double epsilon = 1e-8,
            double varianceMomentumSchedule = 0.9999986111120757,
            bool unitGain = true)
        {
            LearningRate = learningRate;
            _momentum = momentum;
            _l1RegularizationWeight = l1RegularizationWeight;
            _l2RegularizationWeight = l2RegularizationWeight;
            _gradientClippingThresholdPerSample = gradientClippingThresholdPerSample;
            _epsilon = epsilon;
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

            var learner = CNTKLib.AdamLearner(
              parameters: new ParameterVector((System.Collections.ICollection)learningParameters),
              learningRateSchedule: new TrainingParameterScheduleDouble(LearningRate, (uint)MinibatchSize),
              momentumSchedule: new TrainingParameterScheduleDouble(_momentum, (uint)MinibatchSize),
              unitGain: true,
              varianceMomentumSchedule: new TrainingParameterScheduleDouble(_varianceMomentumSchedule, (uint)MinibatchSize),
              epsilon: _epsilon,
              adamax: true,
              additionalOptions: learningOptions);

            return learner;
        }
    }
}

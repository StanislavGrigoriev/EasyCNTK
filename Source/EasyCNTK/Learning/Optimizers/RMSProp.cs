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
    /// Оптимизатор RMSProp. Аналог <seealso cref="AdaDelta"/>. Первоисточник: http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf
    /// </summary>
    public sealed class RMSProp : Optimizer
    {
        private double _l1RegularizationWeight;
        private double _l2RegularizationWeight;
        private double _gradientClippingThresholdPerSample;
        private double _gamma;
        private double _inc;
        private double _dec;
        private double _max;
        private double _min;
        public override double LearningRate { get; }
        public override int MinibatchSize { get; set; }

        /// <summary>
        /// Инициализирует оптимизатор RMSProp 
        /// </summary>
        /// <param name="learningRate">Скорость обучения</param>
        /// <param name="minibatchSize">Размер минипакета, требуется CNTK чтобы масштабировать параметры оптимизатора для более эффективного обучения. Если равен 0, то будет использован размер митибатча при обучении.</param>
        /// <param name="gamma">Коэффициент передачи для предыдущего градиента. Должен быть в пределах [0;1]</param>
        /// <param name="increment">Коэффициент увеличения скорости обучения. Должен быть больше 1. По умолчанию увеличение на 5%</param>
        /// <param name="decrement">Коэффициент уменьшения скорости обучения. Должен быть в пределах [0;1]. По умолчанию уменьшение на 5%</param>
        /// <param name="max">Максимальная скорость обучения. Должна быть больше 0 и min</param>
        /// <param name="min">Минимальная скорость обучения. Должна быть больше 0 и меньше max</param>
        /// <param name="l1RegularizationWeight">Коэффициент L1 нормы, если 0 - регуляризация не применяется</param>
        /// <param name="l2RegularizationWeight">Коэффициент L2 нормы, если 0 - регуляризация не применяется</param>
        /// <param name="gradientClippingThresholdPerSample">Порог отсечения градиента на каждый пример обучения, используется преимущественно для борьбы с взрывным градиентом в глубоких реккурентных сетях.
        /// По умолчанию установлен в <seealso cref="double.PositiveInfinity"/> - отсечение не используется. Для использования установите необходимый порог.</param>
        public RMSProp(double learningRate,
            int minibatchSize = 0,
            double gamma = 0.95,
            double increment = 1.05,
            double decrement = 0.95,
            double max = 0.2,
            double min = 1e-08,            
            double l1RegularizationWeight = 0,
            double l2RegularizationWeight = 0,
            double gradientClippingThresholdPerSample = double.PositiveInfinity)
        {
            LearningRate = learningRate;
            _gamma = gamma;
            _inc = increment;
            _dec = decrement;
            _max = max;
            _min = min;
            MinibatchSize = minibatchSize;
            _l1RegularizationWeight = l1RegularizationWeight;
            _l2RegularizationWeight = l2RegularizationWeight;
            _gradientClippingThresholdPerSample = gradientClippingThresholdPerSample;
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
            return CNTKLib.RMSPropLearner(new ParameterVector((ICollection)learningParameters),
                new TrainingParameterScheduleDouble(LearningRate, (uint)MinibatchSize),
                _gamma,
                _inc,
                _dec,
                _max,
                _min,
                true,
                learningOptions);
        }
    }
}

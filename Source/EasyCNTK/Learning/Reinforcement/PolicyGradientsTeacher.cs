//
// Copyright (c) Stanislav Grigoriev. All rights reserved.
// grigorievstas9@gmail.com 
// https://github.com/StanislavGrigoriev/EasyCNTK
//
// Licensed under the MIT license. See LICENSE.txt file in the project root for full license information.
//
using CNTK;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace EasyCNTK.Learning.Reinforcement
{
    /// <summary>
    /// Реализует механизм обучения методом Policy Gradients
    /// </summary>
    /// <typeparam name="T"></typeparam>
    public class PolicyGradientsTeacher<T> : AgentTeacher<T> where T : IConvertible
    {
        public PolicyGradientsTeacher(Environment environment, DeviceDescriptor device) : base(environment, device) { }

        /// <summary>
        /// Обучает агента, модель которого представлена сетью прямого распространения (не рекуррентной). Используется в случае когда модель оперирует только текущим состоянием среды, не учитывая предыдущие состояния.
        /// </summary>
        /// <param name="agent">Агент для обучения, сеть заданной архитектуры</param>
        /// <param name="iterationCount">Количество итераций обучения (эпох)</param>
        /// <param name="rolloutCount">Количество прогонов(в случае игры - прохождений уровня до окончания игры <seealso cref="Environment.IsTerminated"/>), которое будет пройдено прежде чем обновятся веса.
        /// Можно интерпретировать как количество обучающих данных на одну эпоху.</param>
        /// <param name="minibatchSize">Размер минибатча для обучения</param>
        /// <param name="actionPerIteration">Произвольное действие, которое требуется выполнять каждую эпоху. Позволяет прервать процесс тренировки. Входные параметры: эпоха, loss-ошибка, evaluation-ошибка. 
        /// Выходные: true - прервать процесс тренировки, false - продолжить тренировку.
        /// Используется для осуществления логирования, отображения процесса обучения, сохранения промежуточных чекпоинтов модели и т.п.</param>
        /// <param name="gamma">Коэффициент затухания награды(reward), при вычислении Discounted reward</param>
        /// <returns></returns>
        public Sequential<T> Teach(Sequential<T> agent, int iterationCount, int rolloutCount, int minibatchSize, Func<int, double, double, bool> actionPerIteration = null, double gamma = 0.99)
        {
            for (int iteration = 0; iteration < iterationCount; iteration++)
            {
                var data = new LinkedList<(int rollout, int actionNumber, T[] state, T[] action, T reward)>();
                for (int rolloutNumber = 0; rolloutNumber < rolloutCount; rolloutNumber++)
                {
                    int actionNumber = 0;
                    while (!Environment.IsTerminated)
                    {
                        var currentState = Environment.GetCurrentState<T>();
                        var action = agent.Predict(currentState, Device);
                        var reward = Environment.PerformAction(action);
                        data.AddLast((rolloutNumber, ++actionNumber, currentState, action, reward));
                    }
                    Environment.Reset();
                }
                var discountedRewards = new T[data.Count];
                foreach (var rollout in data.GroupBy(p => p.rollout))
                {
                    var steps = rollout.ToList();
                    steps.Sort((a, b) => a.actionNumber > b.actionNumber ? 1 : a.actionNumber < b.actionNumber ? -1 : 0); //во возрастанию actionNumber
                    for (int i = 0; i < steps.Count; i++)
                    {
                        var remainingRewards = steps.GetRange(i, steps.Count - i)
                            .Select(p => Environment.HasRewardOnlyForRollout ? steps[steps.Count - 1].reward : p.reward)
                            .ToArray();
                        discountedRewards[i] = CalculateDiscountedReward(remainingRewards, gamma);
                    }
                }

                var features = data.Select(p => p.state);
                var labels = data.Zip(discountedRewards, (d, reward) => Multiply(d.action, reward));
                var dataset = features.Zip(labels, (f, l) => f.Concat(l).ToArray()).ToArray();
                var inputDim = features.FirstOrDefault().Length;

                var fitResult = agent.Fit(dataset,
                                        inputDim,
                                        minibatchSize,
                                        GetLoss()[0],
                                        GetEvalLoss()[0],
                                        GetOptimizer()[0],
                                        1,
                                        false,
                                        Device);
                data.Clear();
                var needStop = actionPerIteration?.Invoke(iteration, fitResult.LossError, fitResult.EvaluationError);
                if (needStop.HasValue && needStop.Value)
                    break;
            }
            return agent;
        }
        /// <summary>
        /// Обучает агента, модель которого представлена рекуррентной сетью. Используется в случае когда модель оперирует цепочкой состояний среды.
        /// </summary>
        /// <param name="agent">Агент для обучения, сеть заданной архитектуры</param>
        /// <param name="iterationCount">Количество итераций обучения (эпох)</param>
        /// <param name="rolloutCount">Количество прогонов(в случае игры - прохождений уровня до окончания игры <seealso cref="Environment.IsTerminated"/>), которое будет пройдено прежде чем обновятся веса.
        /// Можно интерпретировать как количество обучающих данных на одну эпоху.</param>
        /// <param name="minibatchSize">Размер минибатча для обучения</param>
        /// <param name="sequenceLength">Длина последовательности: цепочка из предыдущих состояних среды на каждом действии.</param>
        /// <param name="actionPerIteration">Произвольное действие, которое требуется выполнять каждую эпоху. Позволяет прервать процесс тренировки. Входные параметры: эпоха, loss-ошибка, evaluation-ошибка. 
        /// Выходные: true - прервать процесс тренировки, false - продолжить тренировку.
        /// Используется для осуществления логирования, отображения процесса обучения, сохранения промежуточных чекпоинтов модели и т.п.</param>
        /// <param name="gamma">Коэффициент затухания награды(reward), при вычислении Discounted reward</param>
        /// <returns></returns>
        public Sequential<T> Teach(Sequential<T> agent, int iterationCount, int rolloutCount, int minibatchSize, int sequenceLength, Func<int, double, double, bool> actionPerIteration = null, double gamma = 0.99)
        {
            for (int iteration = 0; iteration < iterationCount; iteration++)
            {
                var data = new List<(int rollout, int actionNumber, T[] state, T[] action, T reward)>();
                for (int rolloutNumber = 0; rolloutNumber < rolloutCount; rolloutNumber++)
                {
                    int actionNumber = 0;
                    while (!Environment.IsTerminated)
                    {
                        var currentState = Environment.GetCurrentState<T>();
                        var sequence = actionNumber < sequenceLength
                            ? data.GetRange(data.Count - actionNumber, actionNumber)
                            : data.GetRange(data.Count - sequenceLength - 1, sequenceLength - 1);
                        var sequenceStates = sequence
                            .Select(p => p.state)
                            .ToList();
                        sequenceStates.Add(currentState);
                        var action = agent.Predict(sequenceStates, Device);
                        var reward = Environment.PerformAction(action);
                        data.Add((rolloutNumber, ++actionNumber, currentState, action, reward));
                    }
                    Environment.Reset();
                }
                var discountedRewards = new T[data.Count];
                foreach (var rollout in data.GroupBy(p => p.rollout))
                {
                    var steps = rollout.ToList();
                    for (int i = 0; i < steps.Count; i++)
                    {
                        var remainingRewards = steps.GetRange(i, steps.Count - i)
                            .Select(p => Environment.HasRewardOnlyForRollout ? steps[steps.Count - 1].reward : p.reward)
                            .ToArray();
                        discountedRewards[i] = CalculateDiscountedReward(remainingRewards, gamma);
                    }
                }

                var features = new List<IList<T[]>>();
                var labels = new List<T[]>();
                var dataWithDiscountedReward = data.Zip(discountedRewards, (dat, reward) => (dat, reward)).GroupBy(p => p.dat.rollout);
                foreach (var rollout in dataWithDiscountedReward)
                {
                    var steps = rollout.ToList();
                    steps.Sort((a, b) => a.dat.actionNumber > b.dat.actionNumber ? 1 : a.dat.actionNumber < b.dat.actionNumber ? -1 : 0); //во возрастанию actionNumber
                    for (int i = 0; i < steps.Count; i++)
                    {
                        if (i < sequenceLength)
                        {
                            features.Add(steps.GetRange(0, i + 1).Select(p => p.dat.state).ToArray());
                        }
                        else
                        {
                            features.Add(steps.GetRange(i - sequenceLength, sequenceLength).Select(p => p.dat.state).ToArray());
                        }
                        labels.Add(Multiply(steps[i].dat.action, steps[i].reward));
                    }
                }

                var fitResult = agent.Fit(features,
                                        labels,
                                        minibatchSize,
                                        GetLoss()[0],
                                        GetEvalLoss()[0],
                                        GetOptimizer()[0],
                                        1,
                                        Device);
                data.Clear();
                var needStop = actionPerIteration?.Invoke(iteration, fitResult.LossError, fitResult.EvaluationError);
                if (needStop.HasValue && needStop.Value)
                    break;
            }
            return agent;
        }

    }
}

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
using System.Globalization;
using System.Linq;
using System.Text;

namespace EasyCNTK.Learning.Reinforcement
{
    /// <summary>
    /// Реализует механизм обучения методом Actor Critic
    /// </summary>
    /// <typeparam name="T"></typeparam>
    public class ActorCriticTeacher<T> : AgentTeacher<T> where T : IConvertible
    {
        public ActorCriticTeacher(Environment environment, DeviceDescriptor device) : base(environment, device) { }

        /// <summary>
        /// Обучает агента, модель которого представлена сетью прямого распространения (не рекуррентной) с двумя выходами(не путать с размерностью выхода). Используется в случае когда модель оперирует только текущим состоянием среды, не учитывая предыдущие состояния.
        /// </summary>
        /// <param name="agent">Агент для обучения, сеть заданной архитектуры с двумя выходами: 1 выход - действия агента, 2 выход - средняя награда при выполнении действия(одно число)</param>
        /// <param name="iterationCount">Количество итераций обучения (эпох)</param>
        /// <param name="rolloutCount">Количество прогонов(в случае игры - прохождений уровня до окончания игры <seealso cref="Environment.IsTerminated"/>), которое будет пройдено прежде чем обновятся веса.
        /// Можно интерпретировать как количество обучающих данных на одну эпоху.</param>
        /// <param name="minibatchSize">Размер минибатча для обучения</param>
        /// <param name="actionPerIteration">Произвольное действие, которое требуется выполнять каждую эпоху. Позволяет прервать процесс тренировки. Входные параметры: эпоха, loss-ошибка, evaluation-ошибка. 
        /// Выходные: true - прервать процесс тренировки, false - продолжить тренировку.
        /// Используется для осуществления логирования, отображения процесса обучения, сохранения промежуточных чекпоинтов модели и т.п.</param>
        /// <param name="gamma">Коэффициент затухания награды(reward), при вычислении Discounted reward</param>
        /// <param name="epsilon">Величина, на которую должны отличаться два вещественных числа, чтобы считаться разными. Необходимо для вычисления похожих состояний среды.</param>
        /// <returns></returns>
        public SequentialMultiOutput<T> Teach(SequentialMultiOutput<T> agent, int iterationCount, int rolloutCount, int minibatchSize, Func<int, double[], double[], bool> actionPerIteration = null, double gamma = 0.99, double epsilon = 0.01)
        {
            if (agent.Model.Outputs.Count != 2)
                throw new NotSupportedException("Количество выходов(ветвей) агента должно быть равно 2. Другие конфигурации не поддерживаются.");
            if (agent.Model.Outputs[1].Shape.Rank != 1 || agent.Model.Outputs[1].Shape.Dimensions[0] != 1)
                throw new NotSupportedException("Размерность второго выхода агента должна быть равна 1(выход должен возвращать одно число). Другие конфигурации не поддерживаются.");

            for (int iteration = 0; iteration < iterationCount; iteration++)
            {
                var data = new LinkedList<(int rollout, int actionNumber, T[] state, T[] action, T reward, T agentReward)>();
                for (int rolloutNumber = 0; rolloutNumber < rolloutCount; rolloutNumber++)
                {
                    int actionNumber = 0;
                    while (!Environment.IsTerminated)
                    {
                        var currentState = Environment.GetCurrentState<T>();
                        var agentOutput = agent.Predict(currentState, Device);
                        var action = agentOutput[0];
                        var agentReward = agentOutput[1][0];
                        var reward = Environment.PerformAction(action);
                        data.AddLast((rolloutNumber, ++actionNumber, currentState, action, reward, agentReward));
                    }
                    Environment.Reset();
                }
                //1 - сначала посчитать средний reward для каждого состояния = baseline, присвоить состоянию его средний reward(который отдает Environment - baseline) - это будут метки для обучения второй головы
                var baselines = data
                    .GroupBy(p => p.state, new TVectorComparer<T>(epsilon))
                    .ToDictionary(p => p.Key, q => q.Average(z => z.reward.ToDouble(CultureInfo.InvariantCulture)), new TVectorComparer<T>(epsilon));
                var baselineRewards = data
                    .Select(p => baselines[p.state])
                    .ToArray();

                //2 - потом посчитать уже advatageReward по формуле: Yi*(reward - agentReward) - это будут метки для обучения первой головы
                var elementTypeCode = data.First.Value.reward.GetTypeCode();
                var advantageReward = new T[data.Count];
                foreach (var rollout in data.GroupBy(p => p.rollout))
                {
                    var steps = rollout.ToList();
                    steps.Sort((a, b) => a.actionNumber > b.actionNumber ? 1 : a.actionNumber < b.actionNumber ? -1 : 0); //во возрастанию actionNumber
                    for (int i = 0; i < steps.Count; i++)
                    {
                        var remainingRewards = steps.GetRange(i, steps.Count - i)
                            .Select(p => Environment.HasRewardOnlyForRollout
                                ? steps[steps.Count - 1].reward.ToDouble(CultureInfo.InvariantCulture) - steps[i].agentReward.ToDouble(CultureInfo.InvariantCulture)
                                : p.reward.ToDouble(CultureInfo.InvariantCulture) - p.agentReward.ToDouble(CultureInfo.InvariantCulture))
                            .Select(p => (T)Convert.ChangeType(p, elementTypeCode))
                            .ToArray();

                        advantageReward[i] = CalculateDiscountedReward(remainingRewards, gamma);
                    }
                }

                var features = data.Select(p => p.state).ToArray();
                var actionLabels = data.Zip(advantageReward, (d, reward) => Multiply(d.action, reward));
                var labels = actionLabels.Zip(baselineRewards, (action, baseline) => new[] { action, new T[] { (T)Convert.ChangeType(baseline, elementTypeCode) } }).ToArray();

                var fitResult = agent.Fit(features,
                                        labels,
                                        minibatchSize,
                                        GetLoss(),
                                        GetEvalLoss(),
                                        GetOptimizer(),
                                        1,
                                        false,
                                        Device);
                data.Clear();
                var needStop = actionPerIteration?.Invoke(iteration, fitResult.Select(p => p.LossError).ToArray(), fitResult.Select(p => p.EvaluationError).ToArray());
                if (needStop.HasValue && needStop.Value)
                    break;
            }
            return agent;
        }
        /// <summary>
        /// Обучает агента, модель которого представлена рекуррентной сетью с двумя выходами(не путать с размерностью выхода). Используется в случае когда модель оперирует цепочкой состояний среды.
        /// </summary>
        /// <param name="agent">Агент для обучения, сеть заданной архитектуры с двумя выходами: 1 выход - действия агента, 2 выход - средняя награда при выполнении действия(одно число)</param>
        /// <param name="iterationCount">Количество итераций обучения (эпох)</param>
        /// <param name="rolloutCount">Количество прогонов(в случае игры - прохождений уровня до окончания игры <seealso cref="Environment.IsTerminated"/>), которое будет пройдено прежде чем обновятся веса.
        /// Можно интерпретировать как количество обучающих данных на одну эпоху.</param>
        /// <param name="minibatchSize">Размер минибатча для обучения</param>
        /// <param name="sequenceLength">Длина последовательности: цепочка из предыдущих состояних среды на каждом действии.</param>
        /// <param name="actionPerIteration">Произвольное действие, которое требуется выполнять каждую эпоху. Позволяет прервать процесс тренировки. Входные параметры: эпоха, loss-ошибка, evaluation-ошибка. 
        /// Выходные: true - прервать процесс тренировки, false - продолжить тренировку.
        /// Используется для осуществления логирования, отображения процесса обучения, сохранения промежуточных чекпоинтов модели и т.п.</param>
        /// <param name="gamma">Коэффициент затухания награды(reward), при вычислении Discounted reward</param>
        /// <param name="epsilon">Величина, на которую должны отличаться два вещественных числа, чтобы считаться разными. Необходимо для вычисления похожих состояний среды.</param>
        /// <returns></returns>
        public SequentialMultiOutput<T> Teach(SequentialMultiOutput<T> agent, int iterationCount, int rolloutCount, int minibatchSize, int sequenceLength, Func<int, double[], double[], bool> actionPerIteration = null, double gamma = 0.99, double epsilon = 0.01)
        {
            if (agent.Model.Outputs.Count != 2)
                throw new NotSupportedException("Количество выходов(ветвей) агента должно быть равно 2. Другие конфигурации не поддерживаются.");
            if (agent.Model.Outputs[1].Shape.Rank != 1 || agent.Model.Outputs[1].Shape.Dimensions[0] != 1)
                throw new NotSupportedException("Размерность второго выхода агента должна быть равна 1(выход должен возвращать одно число). Другие конфигурации не поддерживаются.");

            for (int iteration = 0; iteration < iterationCount; iteration++)
            {
                var data = new List<(int rollout, int actionNumber, T[] state, T[] action, T reward, T agentReward)>();
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
                        var agentOutput = agent.Predict(sequenceStates, Device);
                        var action = agentOutput[0];
                        var agentReward = agentOutput[1][0];
                        var reward = Environment.PerformAction(action);
                        data.Add((rolloutNumber, ++actionNumber, currentState, action, reward, agentReward));
                    }
                    Environment.Reset();
                }
                //1 - сначала посчитать средний reward для каждого состояния = baseline, присвоить состоянию его средний reward(который отдает Environment - baseline) - это будут метки для обучения второй головы
                var baselines = data
                    .GroupBy(p => p.state, new TVectorComparer<T>(epsilon))
                    .ToDictionary(p => p.Key, q => q.Average(z => z.reward.ToDouble(CultureInfo.InvariantCulture)), new TVectorComparer<T>(epsilon));
                var baselineRewards = data
                    .Select(p => baselines[p.state])
                    .ToArray();

                //2 - потом посчитать уже advatageReward по формуле: Yi*(reward - agentReward) - это будут метки для обучения первой головы
                var elementTypeCode = data[0].reward.GetTypeCode();
                var advantageReward = new T[data.Count];
                foreach (var rollout in data.GroupBy(p => p.rollout))
                {
                    var steps = rollout.ToList();
                    steps.Sort((a, b) => a.actionNumber > b.actionNumber ? 1 : a.actionNumber < b.actionNumber ? -1 : 0); //во возрастанию actionNumber
                    for (int i = 0; i < steps.Count; i++)
                    {
                        var remainingRewards = steps.GetRange(i, steps.Count - i)
                            .Select(p => Environment.HasRewardOnlyForRollout
                                ? steps[steps.Count - 1].reward.ToDouble(CultureInfo.InvariantCulture) - steps[i].agentReward.ToDouble(CultureInfo.InvariantCulture)
                                : p.reward.ToDouble(CultureInfo.InvariantCulture) - p.agentReward.ToDouble(CultureInfo.InvariantCulture))
                            .Select(p => (T)Convert.ChangeType(p, elementTypeCode))
                            .ToArray();

                        advantageReward[i] = CalculateDiscountedReward(remainingRewards, gamma);
                    }
                }

                var features = new List<IList<T[]>>();
                var labels = new List<T[][]>();
                var dataWithAdvantageRewardAndBaselines = data
                    .Zip(advantageReward, (dat, reward) => (dat, reward))
                    .Zip(baselineRewards, (first, baseline) => (first.dat, first.reward, baseline))
                    .GroupBy(p => p.dat.rollout);
                foreach (var rollout in dataWithAdvantageRewardAndBaselines)
                {
                    var steps = rollout.ToList();
                    steps.Sort((a, b) => a.dat.actionNumber > b.dat.actionNumber ? 1 : a.dat.actionNumber < b.dat.actionNumber ? -1 : 0); //во возрастанию actionNumber
                    for (int i = 0; i < steps.Count; i++)
                    {
                        if (i < sequenceLength)
                        {
                            features.Add(steps
                                .GetRange(0, i + 1)
                                .Select(p => p.dat.state)
                                .ToArray());
                        }
                        else
                        {
                            features.Add(steps
                                .GetRange(i - sequenceLength, sequenceLength)
                                .Select(p => p.dat.state)
                                .ToArray());
                        }
                        labels.Add(new[]
                        {
                            Multiply(steps[i].dat.action, steps[i].reward),
                            new T[] { (T)Convert.ChangeType(steps[i].baseline, elementTypeCode) }
                        });
                    }
                }

                var fitResult = agent.Fit(features,
                                        labels,
                                        minibatchSize,
                                        GetLoss(),
                                        GetEvalLoss(),
                                        GetOptimizer(),
                                        1,
                                        false,
                                        Device);
                data.Clear();
                var needStop = actionPerIteration?.Invoke(iteration, fitResult.Select(p => p.LossError).ToArray(), fitResult.Select(p => p.EvaluationError).ToArray());
                if (needStop.HasValue && needStop.Value)
                    break;
            }
            return agent;
        }

    }
}

using CNTK;
using System;
using System.Collections.Generic;
using System.Text;
using System.Linq;
using System.Globalization;
using EasyCNTK.LossFunctions;
using EasyCNTK.Learning.Optimizers;
using EasyCNTK.Learning;

namespace EasyCNTK.Learning.Reinforcement
{
    public class ForwardAgentTeacher<T> : AgentTeacher<T> where T:IConvertible
    {   
        public ForwardAgentTeacher(Environment environment, DeviceDescriptor device): base(environment, device) { }       

        public Sequential<T> TeachByPolicyGradients(Sequential<T> agent, int iterationCount, int rolloutCount, int minibatchSize, Func<int, double, double, bool> actionPerIteration = null, double gamma = 0.99)
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
                        discountedRewards[i] = CalculateDiscountedReward(remainingRewards, gamma) ;
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
        public SequentialMultiOutput<T> TeachByActorCritic(SequentialMultiOutput<T> agent, int iterationCount, int rolloutCount, int minibatchSize, Func<int, double[], double[], bool> actionPerIteration = null, double gamma = 0.99, double epsilon = 0.01)
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
                var labels = actionLabels.Zip(baselineRewards, (action, baseline) => new[] { action, new T[] {(T)Convert.ChangeType(baseline, elementTypeCode) } }).ToArray();

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

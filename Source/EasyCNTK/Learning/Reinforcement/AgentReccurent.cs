using CNTK;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace EasyCNTK.Learning.Reinforcement
{
    public class AgentReccurent<T> : Agent<T> where T:IConvertible
    {
        public AgentReccurent(Environment environment, Sequential<T> model, DeviceDescriptor device) : base(environment, model, device) { }
        public Sequential<T> LearnByPolicyGradients(int iterationCount, int rolloutCount, int minibatchSize, int sequenceLength, Func<int, double, double, Sequential<T>, bool> actionPerIteration = null, double gamma = 0.99)
        {
            for (int iteration = 0; iteration < iterationCount; iteration++)
            {
                var data = new List<(int rollout, int actionNumber, T[] state, T[] action, T reward)>();
                for (int rolloutNumber = 0; rolloutNumber < rolloutCount; rolloutNumber++)
                {
                    int actionNumber = 0;
                    do
                    {
                        var currentState = Environment.GetCurrentState<T>();
                        var sequence = actionNumber < sequenceLength
                            ? data.GetRange(data.Count - actionNumber, actionNumber)
                            : data.GetRange(data.Count - sequenceLength - 1, sequenceLength - 1);
                        var sequenceStates = sequence
                            .Select(p => p.state)
                            .ToList();
                        sequenceStates.Add(currentState);
                        var action = Model.Predict(sequenceStates, Device);
                        var reward = Environment.PerformAction(action);
                        data.Add((rolloutNumber, ++actionNumber, currentState, action, reward));
                    } while (!Environment.IsTerminated);
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
                var dataWithDiscountedReward = data.Zip(discountedRewards, (d, reward) => (d, reward)).GroupBy(p => p.d.rollout);
                foreach (var rollout in dataWithDiscountedReward)
                {
                    var steps = rollout.ToList();
                    for (int i = 0; i < steps.Count; i++)
                    {
                        if (i < sequenceLength)
                        {
                            features.Add(steps.GetRange(0, i + 1).Select(p => p.d.state).ToArray());                            
                        }
                        else
                        {
                            features.Add(steps.GetRange(i - sequenceLength, sequenceLength).Select(p => p.d.state).ToArray());
                        }
                        labels.Add(Multiply(steps[i].d.action, steps[i].reward));
                    }
                }

                var fitResult = Model.Fit(features,
                                        labels,
                                        minibatchSize,
                                        GetLoss(),
                                        GetEvalLoss(),
                                        GetOptimizer(minibatchSize),
                                        1,
                                        Device);
                data.Clear();
                var needStop = actionPerIteration?.Invoke(iteration, fitResult.LossError, fitResult.EvaluationError, Model);
                if (needStop.HasValue && needStop.Value) break;
            }
            return Model;
        }
    }
}

using CNTK;
using System;
using System.Collections.Generic;
using System.Text;
using System.Linq;
using System.Globalization;
using EasyCNTK.LossFunctions;
using EasyCNTK.Learning.Optimizers;

namespace EasyCNTK.Learning.Reinforcement
{
    public class AgentForward<T> : Agent<T> where T:IConvertible
    {      
        public AgentForward(Environment environment, Sequential<T> model, DeviceDescriptor device): base(environment, model, device) { }       

        public Sequential<T> LearnByPolicyGradients(int iterationCount, int rolloutCount, int minibatchSize, Func<int, double, double, Sequential<T>, bool> actionPerIteration = null, double gamma = 0.99)
        {            
            for (int iteration = 0; iteration < iterationCount; iteration++)
            {
                var data = new LinkedList<(int rollout, int actionNumber, T[] state, T[] action, T reward)>();                
                for (int rolloutNumber = 0; rolloutNumber < rolloutCount; rolloutNumber++)
                {
                    int actionNumber = 0;
                    do
                    {
                        var currentState = Environment.GetCurrentState<T>();
                        var action = Model.Predict(currentState, Device);
                        var reward = Environment.PerformAction(action);
                        data.AddLast((rolloutNumber, ++actionNumber, currentState, action, reward));
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

                var features = data.Select(p => p.state);
                var labels = data.Zip(discountedRewards, (d, reward) => Multiply(d.action, reward));
                var dataset = features.Zip(labels, (f, l) => f.Concat(l).ToArray()).ToArray();
                var inputDim = features.FirstOrDefault().Length;
                
                var fitResult = Model.Fit(dataset,
                                        inputDim,
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

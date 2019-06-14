using CNTK;
using System;
using System.Collections.Generic;
using System.Text;

namespace EasyCNTK.Learning.Reinforcement
{
    public abstract class Environment : IDisposable
    {
        public abstract void Dispose();
        public abstract T PerformAction<T>(T[] actionData) where T : IConvertible;
        public abstract T[] GetCurrentState<T>() where T : IConvertible;
        public abstract bool IsTerminated { get; protected set; }
        public abstract void Reset();
        public abstract bool HasRewardOnlyForRollout { get; protected set; }
    }
}

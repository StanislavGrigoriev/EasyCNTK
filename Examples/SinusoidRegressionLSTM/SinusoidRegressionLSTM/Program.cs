using CNTK;
using System;
using System.Collections.Generic;
using System.Linq;
using EasyCNTK;
using EasyCNTK.ActivationFunctions;
using EasyCNTK.Layers;
using EasyCNTK.Learning;
using EasyCNTK.Learning.Metrics;
using EasyCNTK.Learning.Optimizers;
using EasyCNTK.LossFunctions;

namespace SinusoidRegressionLSTM
{
    class Program
    {
        static void Main(string[] args)
        {
            CNTKLib.SetFixedRandomSeed(0); //для воспроизводимости. т.к. инициализация весов в слоях нейросети
                                           //зависит от генератора случайных чисел CNTK

            //создаем симулированный датасет из последовательностей описывающих синусоиду
            var dataset = Enumerable.Range(1, 2000)
                .Select(p => Math.Sin(p / 100.0)) //уменьшаем шаг, чтобы синусоида была плавнее
                .Segment(10) //разбиваем синусоиду на сегменты по 10 элементов
                .Select(p => (featureSequence: p.Take(9).Select(q => new[] { q }).ToArray(), //задаем последовательность из 9 элементов, каждый элемент размерности 1 (может быть: 1, 2, 3...n)
                                        label: new[] { p[9] })) //задаем метку для последовательности размерности 1 (может быть: 1, 2, 3...n)
                .ToArray();
            dataset.Split(0.7, out var train, out var test);

            int minibatchSize = 16;
            int epochCount = 300;
            int inputDimension = 1;
            var device = DeviceDescriptor.GPUDevice(0);

            var model = new Sequential<double>(device, new[] { inputDimension }, inputName: "Input");
            model.Add(new LSTM(1, selfStabilizerLayer: new SelfStabilization()));
            model.Add(new Residual2(1, new Tanh()));

            //можно стыковать слои LSTM друг за другом как в комментарии ниже:
            //var model = new Sequential<double>(device, new[] { inputDimension });
            //model.Add(new Dense(3, new Tanh())); 
            //model.Add(new LSTM(10, isLastLstm: false)); //LSTM так же может быть первым слоем в модели
            //model.Add(new LSTM(5, isLastLstm: false));
            //model.Add(new LSTM(2, selfStabilizerLayer: new SelfStabilization())); 
            //model.Add(new Residual2(1, new Tanh()));

            //используется одна из нескольких перегрузок, которые способны обучать реккурентные сети
            var fitResult = model.Fit(features:     train.Select(p => p.featureSequence).ToArray(), 
                labels:                             train.Select(p => p.label).ToArray(),
                minibatchSize:                      minibatchSize,
                lossFunction:                       new AbsoluteError(),
                evaluationFunction:                 new AbsoluteError(),
                optimizer:                          new Adam(0.005, 0.9, minibatchSize),
                epochCount:                         epochCount,
                device:                             device,
                shuffleSampleInMinibatchesPerEpoch: true,
                ruleUpdateLearningRate: (epoch, learningRate) => learningRate % 50 == 0 ? 0.95 * learningRate : learningRate,
                actionPerEpoch: (epoch, loss, eval) =>
                {
                    Console.WriteLine($"Loss: {loss:F10} Eval: {eval:F3} Epoch: {epoch}");
                    if (loss < 0.05) //критерий остановки достигнут, сохраем модель в файл и заканчиваем обучение (приблизительно на 112 эпохе)
                    {
                        model.SaveModel($"{model}.model", saveArchitectureDescription: false);
                        return true;
                    }
                    return false;
                }, 
                inputName: "Input");

            Console.WriteLine($"Duration train: {fitResult.Duration}");
            Console.WriteLine($"Epochs: {fitResult.EpochCount}");
            Console.WriteLine($"Loss error: {fitResult.LossError}");
            Console.WriteLine($"Eval error: {fitResult.EvaluationError}");

            var metricsTrain = model
                .Evaluate(train.Select(p => p.featureSequence), train.Select(p => p.label), device)
                .GetRegressionMetrics();
            var metricsTest = model
                .Evaluate(test.Select(p => p.featureSequence), test.Select(p => p.label), device)
                .GetRegressionMetrics();
            
            Console.WriteLine($"Train => MAE: {metricsTrain[0].MAE} RMSE: {metricsTrain[0].RMSE} R2: {metricsTrain[0].Determination}");//R2 ~ 0,983
            Console.WriteLine($"Test => MAE: {metricsTest[0].MAE} RMSE: {metricsTest[0].RMSE} R2: {metricsTest[0].Determination}"); //R2 ~ 0,982

            Console.ReadKey();
        }
    }
}

using System;
using System.Collections.Generic;
using System.IO;
using System.IO.Compression;
using System.Linq;
using CNTK;
using EasyCNTK;
using EasyCNTK.ActivationFunctions;
using EasyCNTK.Layers;
using EasyCNTK.Learning;
using EasyCNTK.Learning.Optimizers;
using EasyCNTK.LossFunctions;

namespace MNISTClassifier
{
    class Program
    {
        static double[] MnistOneHotEncoding(int digit)
        {
            if (digit < 0 || digit > 9)
            {
                throw new IndexOutOfRangeException("digit");
            }
            double[] oneHotEncoding = new double[10];
            oneHotEncoding[digit] = 1;
            return oneHotEncoding;
        }

        static void Main(string[] args)
        {
            var datasetTrain = new List<double[]>();
            var datasetTest = new List<double[]>();

            using (var zipToOpen = File.OpenRead(@"Dataset\mnist-in-csv.zip"))
            using (ZipArchive archive = new ZipArchive(zipToOpen, ZipArchiveMode.Read))
            {                   
                using (var train = new StreamReader(archive.GetEntry("mnist_train.csv").Open()))
                {
                    datasetTrain = train.ReadToEnd()
                        .Split('\n')
                        .Skip(1)
                        .Select(p => p.Split(new[] { ',' }, StringSplitOptions.RemoveEmptyEntries))
                        .Where(p => p.Length > 0)
                        .Select(p => p.Select(q => double.Parse(q)).ToArray())                        
                        .Select(p => p.Skip(1) //1 столбец - метка, пропускаем
                            .Concat(MnistOneHotEncoding((int)p[0]))//переносим метку в конец массива с признаками, и сразу кодируем в one-hot-encoding
                            .ToArray())
                        .ToList();
                }
                using (var test = new StreamReader(archive.GetEntry("mnist_test.csv").Open()))
                {
                    datasetTest = test.ReadToEnd()
                        .Split('\n')
                        .Skip(1)
                        .Select(p => p.Split(new[] { ',' }, StringSplitOptions.RemoveEmptyEntries))
                        .Where(p => p.Length > 0)
                        .Select(p => p.Select(q => double.Parse(q)).ToArray())
                        .Select(p => p.Skip(1)
                            .Concat(MnistOneHotEncoding((int)p[0]))
                            .ToArray())
                        .ToList();
                }
            }

            var device = DeviceDescriptor.GPUDevice(0);

            int minibatchSize = 512;
            int inputDimension = 784;
            int epochs = 50;

            var model = new Sequential<double>(device, new[] { inputDimension }, inputName: "Input");
            model.Add(new Residual2(784, new Tanh()));                       
            model.Add(new Residual2(300, new Tanh()));
            model.Add(new Dense(10, new Sigmoid()));

            var fitResult = model.Fit(datasetTrain,
                inputDimension,
                minibatchSize,
                new SquaredError(),
                new ClassificationError(),
                new Adam(0.1, 0.9, minibatchSize),
                epochs,                
                shuffleSampleInMinibatchesPerEpoch: false,
                device: device,
                ruleUpdateLearningRate: (epoch, learningRate) => epoch % 10 == 0 ? 0.95 * learningRate : learningRate,
                actionPerEpoch: (epoch, loss, eval) =>
                {
                    Console.WriteLine($"Loss: {loss:F10} Eval: {eval:F3} Epoch: {epoch}");
                    if (eval < 0.05) //ошибка классфикации меньше 5%, сохраем модель в файл и заканчиваем обучение
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
                .Evaluate(datasetTrain, inputDimension, device)
                .GetOneLabelClassificationMetrics();

            Console.WriteLine($"---Train---");
            Console.WriteLine($"Accuracy: {metricsTrain.Accuracy}");
            metricsTrain.ClassesDistribution.ForEach(p => Console.WriteLine($"Class: {p.Index} | Precision: {p.Precision:F5} | Recall: {p.Recall:F5} | Fraction: {p.Fraction * 100:F3}"));

            var metricsTest = model
                .Evaluate(datasetTest, inputDimension, device)
                .GetOneLabelClassificationMetrics();            
  
            Console.WriteLine($"---Test---");
            Console.WriteLine($"Accuracy: {metricsTest.Accuracy}");
            metricsTest.ClassesDistribution.ForEach(p => Console.WriteLine($"Class: {p.Index} | Precision: {p.Precision:F5} | Recall: {p.Recall:F5} | Fraction: {p.Fraction * 100:F3}"));

            Console.Read();
        }
    }
}

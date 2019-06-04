# EasyCNTK
EasyCNTK - C# библиотека для глубокого обучения. Является оберткой над C# CNTK API, позволяет конфигурировать и обучать нейронные сети используя более высокоуровневые абстракции, похожие на Keras. Содержит реализации слоев, оптимизаторов, функций потерь, метрики для оценки качества классификации и регрессии, а так же вспомогательные методы для вычисления статистики датасета, сбалансированного разбиения и т.п. Библиотека ориентирована на пользователей имеющих некоторые знания в глубоком обучении и отдаленно знакомых с C# CNTK API, но желающих тренировать нейросети из кода на C#.
## Быстрый старт
__1__. Установить NuGet пакет EasyCNTK:
```
PM> Install-Package EasyCNTK -Version 0.2.0
```
Вместе с ним будет установлен `CNTK.GPU v2.7` - который позволяет производить обучение на CPU и GPU. В дальнейшем будет сборка для `CPUOnly` проектов. EasyCNTK написан под `.net Standart 2.0`.  
__2__. Переключить проект (и все используемые им проекты/сборки) в конфигурацию платформы `x64`. Без этого работать ничего не будет, поскольку `CNTK` реализован только на платформе `x64`.  
__3__. Добавить директивы Using: 
```
using CNTK;
using EasyCNTK;
using EasyCNTK.ActivationFunctions;
using EasyCNTK.Layers;
using EasyCNTK.Learning;
using EasyCNTK.Learning.Metrics;
using EasyCNTK.Learning.Optimizers;
using EasyCNTK.LossFunctions;
```
__4__. Основное использование на примере одноклассовой классификации:
```
        {
            List<double[]> dataset = LoadMyDataset();//вначале массива признаки, затем метки классов
            dataset.SplitBalanced(0.7, //разбивает датасет на 2 части в заданном соотношении, сохраняя одинаковое распределение классов в обоих частях
                sample => sample.Last(), //определение метки классов вычисления их распределения 
                out var datasetTrain,
                out var datasetTest,
                randomizeSplit: true);

            List<FeatureStatistic> statistic = dataset.ComputeStatisticForCollection(); //Статистика для каждого признака (среднее, мин, макс, дисперсия и т.п.) 

            var device = DeviceDescriptor.GPUDevice(0); //указали что хотим обучать на GPU
            int minibatchSize = 512;
            int inputDimension = 784;
            int epochs = 50;

            //создание модели. Поддерживаются float и double веса нейросети
            var model = new Sequential<double>(device, new[] { inputDimension });
            model.Add(new Residual2(784, new Tanh()));
            model.Add(new Residual2(300, new Tanh()));
            model.Add(new Dense(10, new Sigmoid()));

            //обучение модели. Поддерживается множество перегрузок, для разных вариантов обучения
            var fitResult = model.Fit(
                trainData:          datasetTrain,
                inputDim:           inputDimension,
                minibatchSize:      minibatchSize,
                lossFunction:       new SquaredError(),
                evaluationFunction: new ClassificationError(),
                optimizer:          new Adam(0.1, 0.9, minibatchSize),
                epochCount:         epochs,
                device:             device,
                shuffleSampleInMinibatchesPerEpoch: false,
                ruleUpdateLearningRate: (epoch, learningRate) => epoch % 10 == 0 ? 0.95 * learningRate : learningRate,
                actionPerEpoch: (epoch, loss, eval) => //Вход: эпоха, ошибка loss, ошибка eval. Выход: true - закончить обучение, false - продолжить.
                {
                    //необязательный метод, используется для произвольных действий пользователя

                    Console.WriteLine($"Loss: {loss:F10} Eval: {eval:F3} Epoch: {epoch}"); //например отображаем прогресс обучения
                    if (eval < 0.05) //или ранняя остановка: если ошибка классфикации меньше 5%, сохраем модель в файл и заканчиваем обучение
                    {
                        model.SaveModel($"{model}.model", saveArchitectureDescription: false);
                        return true;
                    }
                    return false;
                });

            TimeSpan duration      = fitResult.Duration;//продолжительность обучения
            int epochCount         = fitResult.EpochCount;//количесво эпох обучения
            double lossError       = fitResult.LossError;//ошибка функции потерь
            double evaluationError = fitResult.EvaluationError;//ошибка оценочной функции

            OneLabelClassificationMetrics metricsTest = model
                    .Evaluate(datasetTest, inputDimension, device) //выполняем оценку (Predict) для контрольных примеров датасета
                    .GetOneLabelClassificationMetrics(); //вычисляем метрики оценки одноклассовой(многоклассовой/бинарной) классификации(регрессии)
            double accuracy                     = metricsTest.Accuracy;
            double[,] confusionMatrix           = metricsTest.ConfusionMatrix; //матрица ошибок, использует процентное отображение
            List<ClassItem> classesDistribution = metricsTest.ClassesDistribution;
            int classIndex                      = classesDistribution[0].Index; //индекс выхода нейросети, закрепленный за классом
            double classPrecision               = classesDistribution[0].Precision; //точность определения этого класса
            double classFraction                = classesDistribution[0].Fraction; //доля примеров этого класса в контрольной выборке
        }
```
Больше примеров доступно в папке __[Examples](https://github.com/StanislavGrigoriev/EasyCNTK/tree/master/Examples)__.
***
Документация постепенно будет дополняться, то же касается и примеров использования. Со временем добавятся новые реализации слоев(Attention, Convolution3D и т.д.), дополнительных функций расчета статистики и т.п.

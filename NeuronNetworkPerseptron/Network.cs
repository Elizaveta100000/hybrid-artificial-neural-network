using System;
using System.Xml.Linq;

namespace NeuronNetworkPerseptron
{
    public class Network
    {
        /// <summary>
        /// </summary>
        /// <param name="nm">Режима работы сети (Train, Test, Demo).</param>
        /// <param name="numbLayerHiddens">Количество слоев сети.</param>
        /// <param name="numbNeuronLayers">Массив из количества нейронов в слоях.</param>
        public Network(NetworkMode nm, int numbLayerHiddens, int[] numbNeuronLayers, LayerInput LayerInput)
        {
            input_layer = LayerInput;
            hidden_layer = new LayerHidden[numbLayerHiddens];
            for (int i = 0; i < numbLayerHiddens; i++)
            {
                Console.WriteLine($"Слой персептрона {i}...");
                hidden_layer[i] = new LayerHidden(numbNeuronLayers[i + 1], numbNeuronLayers[i],
                    NeuronType.Hidden, $"LayerHidden{i}");
            }
            Console.WriteLine($"Слой персептрона output...");
            output_layer = new LayerOutput(numbNeuronLayers[numbNeuronLayers.Length - 1], numbNeuronLayers[numbNeuronLayers.Length - 2],
                NeuronType.Output, nameof(output_layer));
            fact = new double[numbNeuronLayers[numbNeuronLayers.Length - 1]];
        }
        //все слои сети
        /// <summary>
        /// Входной слой (входной массив).
        /// </summary>
        public LayerInput input_layer = null;
        /// <summary>
        /// Скрытые слои.
        /// </summary>
        public LayerHidden[] hidden_layer = null;
        /// <summary>
        /// Выходной слой.
        /// </summary>
        public LayerOutput output_layer;
        /// <summary>
        /// Массив для хранения выхода сети.
        /// </summary>
        public double[] fact;
        /// <summary>
        /// Обучение сети.
        /// </summary>
        /// <param name="net">Сеть.</param>
        public Network Train(Network net)//backpropagation method
        {
            int epoches = 1500;
            XDocument memory_doc = new XDocument();
            XElement err = new XElement("im");
            XElement err0 = new XElement("Image0");
            XElement err1 = new XElement("Image1");
            XElement err2 = new XElement("Image2");
            for (int k = 0; k < epoches; ++k)
            {
                Console.WriteLine($"Эпоха персептрона {k}...");
                for (int i = 0; i < net.input_layer.Trainset.Length; ++i)
                {
                    //Console.WriteLine($"Картинка {i}");
                    //прямой проход
                    ForwardPass(net.input_layer.Trainset[i].Item1);
                    //вычисление ошибки по итерации
                    double[] errors = new double[net.fact.Length];
                    for (int x = 0; x < errors.Length; ++x)
                    {
                        errors[x] = (x == net.input_layer.Trainset[i].Item2) ? -(net.fact[x] - 1.0d) : -net.fact[x];
                    }
                    if (i == 1)
                        err1.Add(new XElement("error", errors[1].ToString()));
                    else if (i == 0)
                        err0.Add(new XElement("error", errors[0].ToString()));
                    else if (i == 2)
                        err2.Add(new XElement("error", errors[0].ToString()));
                    //обратный проход и коррекция весов
                    double[][] temp_grums = new double[hidden_layer.Length][];
                    temp_grums[0] = net.output_layer.BackwardPass(errors);
                    for (int j = 1; j < temp_grums.Length; j++)
                    {
                        temp_grums[j] = net.hidden_layer[hidden_layer.Length - j].BackwardPass(temp_grums[j - 1]);
                    }
                    net.hidden_layer[0].BackwardPass(temp_grums[temp_grums.Length - 1]);
                }
            }
            err.Add(err0);
            err.Add(err1);
            err.Add(err2);
            memory_doc.Add(err);
            memory_doc.Save("Errors.xml");

            //загрузка скорректированных весов в "память"
            Console.WriteLine("Загрузка скорректированных весов на диск...");
            for (int i = 0; i < hidden_layer.Length; i++)
            {
                net.hidden_layer[i].WeightInitialize(MemoryMode.SET, $"LayerHidden{i}");
            }
            net.output_layer.WeightInitialize(MemoryMode.SET, nameof(output_layer));
            return net;
        }
        /// <summary>
        /// Тестирование сети.
        /// </summary>
        /// <param name="net">Сеть.</param>
        public Network Test(int i)
        {
            ForwardPass(input_layer.Testset[i].Item1);
            if (output_layer.output.output(input_layer, i) != null)
                fact = output_layer.output.output(input_layer, i);
            return this;
        }
        /// <summary>
        /// Прямой проход.
        /// </summary>
        /// <param name="net">Cеть.</param>
        /// <param name="netInput">Массив входных данных.</param>
        public void ForwardPass(double[] netInput)
        {
            hidden_layer[0].Data = netInput;
            for (int i = 0; i < hidden_layer.Length - 1; i++)
            {
                hidden_layer[i].Recognize(this, hidden_layer[i + 1]);
            }
            hidden_layer[hidden_layer.Length - 1].Recognize(null, output_layer);
            output_layer.Recognize(this, null);
        }
    }
}

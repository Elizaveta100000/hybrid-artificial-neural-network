using System;
using System.Xml.Linq;
using NeuronNetworkPerseptron;

namespace NeuronNetworkConvolution
{
    public class Network
    {
        /// <summary>
        /// </summary>
        /// <param name="nm">Режима работы сети (Train, Test, Demo).</param>
        /// <param name="LayerInput">Входной слой.</param>
        public Network(NetworkMode nm, LayerInput LayerInput)
        {
            int[] numbNeuronLayers = { 1800, 120, 84, 2 };
            input_layer = LayerInput;
            Console.WriteLine($"Слой свертки...");
            first_convolution = new ConvolutionFirst();
            Console.WriteLine($"Слой субдескритизации...");
            first_subsamping = new SubsampingFirst();
            Console.WriteLine($"Слой свертки...");
            second_convolution = new ConvolutionSecond();
            Console.WriteLine($"Слой субдескритизации...");
            second_subsamping = new SubsampingSecond();
            int numbLayerHiddens = 2;
            hidden_layer = new LayerHidden[numbLayerHiddens];
            for (int i = 0; i < numbLayerHiddens; i++)
            {
                Console.WriteLine($"Полносвязный слой {i}...");
                hidden_layer[i] = new LayerHidden(numbNeuronLayers[i + 1], numbNeuronLayers[i],
                    NeuronType.Hidden, $"LayerHiddenConvolution{i}");
            }
            Console.WriteLine($"Полносвязный слой output...");
            output_layer = new LayerOutput(numbNeuronLayers[numbNeuronLayers.Length - 1], numbNeuronLayers[numbNeuronLayers.Length - 2],
                NeuronType.Output, "LayerOutputConvolution");
            fact = new double[numbNeuronLayers[numbNeuronLayers.Length - 1]];
        }
        //все слои сети
        /// <summary>
        /// Первый слой свертки.
        /// </summary>
        private ConvolutionFirst first_convolution;
        /// <summary>
        /// Второй слой свертки.
        /// </summary>
        private ConvolutionSecond second_convolution;
        /// <summary>
        /// Первый слой пулинга.
        /// </summary>
        private SubsampingFirst first_subsamping;
        /// <summary>
        /// Второй слой пулинга.
        /// </summary>
        private SubsampingSecond second_subsamping;
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
        public Network Train()//backpropagation method
        {
            int epoches = 1500;
            XDocument memory_doc = new XDocument();
            XElement err = new XElement("im");
            XElement err0 = new XElement("Image0");
            XElement err1 = new XElement("Image1");
            XElement err2 = new XElement("Image2");
            for (int k = 0; k < epoches; ++k)
            {
                Console.WriteLine($"Эпоха свертки {k}...");
                for (int i = 0; i < input_layer.Trainset.Length; ++i)
                {
                    //прямой проход
                    ForwardPass(input_layer.Trainset[i].Item1);
                    //вычисление ошибки по итерации
                    double[] errors = new double[fact.Length];
                    for (int x = 0; x < errors.Length; ++x)
                    {
                        errors[x] = (x == input_layer.Trainset[i].Item2) ? -(fact[x] - 1.0d) : -fact[x];
                    }
                    if (i == 1)
                        err1.Add(new XElement("error", errors[1].ToString()));
                    else if (i == 0)
                        err0.Add(new XElement("error", errors[0].ToString()));
                    else if (i == 2)
                        err2.Add(new XElement("error", errors[0].ToString()));
                    //обратный проход и коррекция весов
                    double[][] temp_grums = new double[4][];
                    temp_grums[0] = output_layer.BackwardPass(errors);
                    temp_grums[1] = hidden_layer[1].BackwardPass(temp_grums[0]);
                    temp_grums[2] = hidden_layer[0].BackwardPass(temp_grums[1]);
                    temp_grums[3] = second_convolution.BackwardPass(temp_grums[2]);
                    first_convolution.BackwardPass(temp_grums[3]);
                }
            }
            err.Add(err0);
            err.Add(err1);
            err.Add(err2);
            memory_doc.Add(err);
            memory_doc.Save("ErrorsConvolution.xml");

            //загрузка скорректированных весов в "память"
            Console.WriteLine("Загрузка скорректированных весов на диск...");
            first_convolution.WeightInitialize(MemoryMode.SET, "ConvolutionFirst");
            second_convolution.WeightInitialize(MemoryMode.SET, "ConvolutionSecond");
            for (int i = 0; i < hidden_layer.Length; i++)
            {
                hidden_layer[i].WeightInitialize(MemoryMode.SET, $"LayerHiddenConvolution{i}");
            }
            output_layer.WeightInitialize(MemoryMode.SET, "LayerOutputConvolution");
            return this;
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
            first_convolution.Data = netInput;
            first_subsamping.Data = first_convolution.Recognize();
            second_convolution.Data = first_subsamping.Recognize();
            second_subsamping.Data = second_convolution.Recognize();
            hidden_layer[0].Data = second_subsamping.Recognize();
            for (int i = 0; i < hidden_layer.Length - 1; i++)
            {
                hidden_layer[i].Recognize(this, hidden_layer[i + 1]);
            }
            hidden_layer[hidden_layer.Length - 1].Recognize(null, output_layer);
            output_layer.Recognize(this, null);
        }
    }
}

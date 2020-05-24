using System.Xml.Linq;
using System;
using NeuronNetworkPerseptron;

namespace NeuronNetworkConvolution
{
    public abstract class Layer
    {
        /// <summary>
        /// Конструктор.
        /// </summary>
        /// <param name="non">Число нейронов текущего слоя.</param>
        /// <param name="nopn">Число нейронов предыдущего слоя.</param>
        /// <param name="nt">Тип нейрона.</param>
        /// <param name="type">Название слоя.</param>
        protected Layer(int non, int nopn, NeuronType nt, string type)
        {
            numofneurons = non;
            numofprevneurons = nopn;
            Neurons = new Neuron[non];
            Console.WriteLine($"Чтение весов");
            double[,] Weights = WeightInitialize(MemoryMode.GET, type);
            Console.WriteLine($"Чтение весов окончено");
            lastdeltaweights = Weights;
            for (int i = 0; i < non; ++i)
            {
                double[] temp_weights = new double[nopn + 1];
                for (int j = 0; j < nopn + 1; ++j)
                {
                    temp_weights[j] = Weights[i, j];
                }
                Neurons[i] = new Neuron(null, temp_weights, nt);
            }
            output = new LayerInput();
        }
        /// <summary>
        /// Число нейронов текущего слоя.
        /// </summary>
        protected int numofneurons;
        /// <summary>
        /// Число нейронов предыдущего слоя.
        /// </summary>
        protected int numofprevneurons;
        /// <summary>
        /// Скорость обучения.
        /// </summary>
        protected const double learningrate = 0.05d;
        /// <summary>
        /// Момент инерции.
        /// </summary>
        protected const double momentum = 0.3d;
        public LayerInput output;
        /// <summary>
        /// Веса предыдущей итерации обучения.
        /// </summary>
        protected double[,] lastdeltaweights;
        /// <summary>
        /// Массив нейронов текущего слоя.
        /// </summary>
        public Neuron[] Neurons { get; set; }
        /// <summary>
        /// Добавление массива нейронов предыдущего слоя. Активация нейронов.
        /// </summary>
        public double[] Data
        {
            set
            {
                for (int i = 0; i < Neurons.Length; ++i)
                {
                    Neurons[i].Inputs = value;
                    Neurons[i].Activator(Neurons[i].Inputs, Neurons[i].Weights);
                }
            }
        }
        /// <summary>
        /// Чтение/запись весовых коэффициентов из/в файл(а).
        /// </summary>
        /// <param name="mm">Режим работы памяти.</param>
        /// <param name="type">азвание слоя.</param>
        /// <returns>Веса слоя.</returns>
        public double[,] WeightInitialize(MemoryMode mm, string type)
        {
            double[,] weights = new double[numofneurons, numofprevneurons + 1];
            XDocument memory_doc = XDocument.Load($@"Resources\{type}_memory.dat");
            long count;
            switch (mm)
            {
                case MemoryMode.GET:
                    count = -1;
                    foreach (XElement el in memory_doc.Element("weights").Elements("weight"))
                    {
                        count++;
                        int k = (int)(count / weights.GetLength(1));
                        weights[k, count - k * weights.GetLength(1)] = Convert.ToDouble(el.Value);
                    }
                    break;
                case MemoryMode.SET:
                    count = -1;
                    foreach(XElement el in memory_doc.Element("weights").Elements("weight"))
                    {
                        count++;
                        int k = (int)(count / weights.GetLength(1));
                        el.Value = Neurons[k].Weights[count - k * weights.GetLength(1)].ToString();
                    }
                    break;
            }
            memory_doc.Save($@"Resources\{type}_memory.dat");
            return weights;
        }
        /// <summary>
        /// Для прямых проходов.
        /// </summary>
        /// <param name="net">Сеть.</param>
        /// <param name="nextLayer">Следующий слой.</param>
        abstract public void Recognize(Network net, Layer nextLayer);
        /// <summary>
        /// Для обратных проходов.
        /// </summary>
        /// <param name="stuff">Массив градиентов.</param>
        /// <returns>бла</returns>
        abstract public double[] BackwardPass(double[] stuff);
    }
}

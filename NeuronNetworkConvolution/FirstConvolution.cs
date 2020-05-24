using System.Xml.Linq;
using System;

namespace NeuronNetworkConvolution
{
    class ConvolutionFirst
    {
        /// <summary>
        /// Конструктор.
        /// </summary>
        public ConvolutionFirst()
        {
            numofneurons = 18816;
            numofprevneurons = 4096;
            Neurons = new ConvolutionOfNeuron[numofneurons];
            double[,] Weights = WeightInitialize(MemoryMode.GET, "ConvolutionFirst");
            Convolutions = new Convolution[6];
            for (int i = 0; i < 6; ++i)
            {
                double[] temp_weights = new double[25];
                for (int j = 0; j < 81; ++j)
                {
                    temp_weights[j] = Weights[i, j];
                }
                Convolutions[i] = new Convolution(temp_weights);
            }
            lastdeltaweights = Weights;
            for (int i = 0; i < numofneurons; i++)
            {
                Neurons[i] = new ConvolutionOfNeuron(null, Convolutions[(int)(i / 3136)]);
            }
        }
        /// <summary>
        /// Число нейронов текущего слоя.
        /// </summary>
        int numofneurons;
        /// <summary>
        /// Число нейронов предыдущего слоя.
        /// </summary>
        int numofprevneurons;
        /// <summary>
        /// Скорость обучения.
        /// </summary>
        const double learningrate = 0.05d;
        /// <summary>
        /// Момент инерции.
        /// </summary>
        const double momentum = 0.3d;
        /// <summary>
        /// Веса предыдущей итерации обучения.
        /// </summary>
        double[,] lastdeltaweights;
        /// <summary>
        /// Массив сверток.
        /// </summary>
        public Convolution[] Convolutions { get; set; }
        /// <summary>
        /// Массив нейронов текущего слоя.
        /// </summary>
        public ConvolutionOfNeuron[] Neurons { get; set; }
        /// <summary>
        /// Входные нейроны.
        /// </summary>
        double[] Inputs { get; set; }
        /// <summary>
        /// Добавление массива нейронов предыдущего слоя. Активация нейронов.
        /// </summary>
        public double[] Data
        {
            set
            {
                if (Inputs == null)
                {
                    Inputs = value;
                    int k = -1;
                    for (int i = 0; i < Neurons.Length; ++i)
                    {
                        k++;
                        int t = ((int)(k / 64)) * 64 - k;
                        if (t == -56)
                            k += 4;
                        if ((int)i / 3136 == (double)i / 3136)
                        {
                            k = 0;
                        }
                        double[] input = new double[81];
                        for (int j = 0; j < 9; j++)
                        {
                            for (int l = 0; l < 9; l++)
                            {
                                input[9 * j + l] = Inputs[k + j * 64 + l];
                            }
                        }
                        Neurons[i].Inputs = input;
                        Neurons[i].Activator();
                    }
                }
                else
                    for (int i = 0; i < Neurons.Length; i++)
                        Neurons[i].Activator();
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
            double[,] weights = new double[6, 81];
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
                    foreach (XElement el in memory_doc.Element("weights").Elements("weight"))
                    {
                        count++;
                        int k = (int)(count / weights.GetLength(1));
                        el.Value = Convolutions[k].Weights[count - k * weights.GetLength(1)].ToString();
                    }
                    break;
            }
            memory_doc.Save($@"Resources\{type}_memory.dat");
            return weights;
        }
        /// <summary>
        /// Прямой проход.
        /// </summary>
        /// <returns>Выход сети.</returns>
        public double[] Recognize()
        {
            double[] hidden_out = new double[Neurons.Length];
            for (int i = 0; i < Neurons.Length; ++i)
                hidden_out[i] = Neurons[i].Output;
            return hidden_out;
        }
        /// <summary>
        /// Для обратных проходов.
        /// </summary>
        /// <param name="gr_sums">Массив градиентов.</param>
        /// <returns>бла</returns>
        public void BackwardPass(double[] gr_sums)
        {
            // увеличиваем градиент в размерах
            double[] gr_sumsThis = new double[numofneurons];
            int k = -2;
            for (int i = 0; i < gr_sums.Length; i++)
            {
                k += 2;
                int t = ((int)(k / 56));
                if (t % 2 != 0)
                    k += 56;
                gr_sumsThis[k] = gr_sums[i];
                gr_sumsThis[k + 1] = gr_sums[i];
                gr_sumsThis[k + 56] = gr_sums[i];
                gr_sumsThis[k + 57] = gr_sums[i];
            }
            for (int i = 0; i < numofneurons; ++i)
                for (int n = 0; n < 81; ++n)
                {
                   double deltaw = (momentum * lastdeltaweights[(int)(i / 3136), n] + learningrate 
                        * Neurons[i].Inputs[n] * Neurons[i].Derivative  * gr_sumsThis[i]);
                    if ((double)(i / 3136) == (int)(i / 3136))
                        lastdeltaweights[(int)(i / 3136), n] = deltaw;
                    Convolutions[(int)(i / 3136)].Weights[n] += deltaw;//коррекция весов
                }
            for (int i = 0; i < numofneurons; i++)
            {
                Neurons[i].Conv = Convolutions[(int)(i / 3136)];
            }
        }
    }
}

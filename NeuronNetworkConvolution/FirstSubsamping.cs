
namespace NeuronNetworkConvolution
{
    class SubsampingFirst
    {
        public SubsampingFirst()
        {
            numofneurons = 4704;
            numofprevneurons = 18816;
            Neurons = new SubsampingNeuron[numofneurons];
            for (int i = 0; i < numofneurons; i++)
            {
                Neurons[i] = new SubsampingNeuron(null);
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
        /// Массив нейронов текущего слоя.
        /// </summary>
        public SubsampingNeuron[] Neurons { get; set; }
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
                Inputs = value;
                int k = -2;
                for (int i = 0; i < Neurons.Length; ++i)
                {
                    double[] input = new double[4];
                    k += 2;
                    int t = ((int)(k / 56));
                    if (t % 2 != 0)
                        k += 56;
                    if ((int)i / 784 == (double)i / 784)
                    {
                        k = 0;
                    }
                    input[0] = Inputs[k];
                    input[1] = Inputs[k + 1];
                    input[2] = Inputs[k + 56];
                    input[3] = Inputs[k + 57];
                    Neurons[i].Inputs = input;
                    Neurons[i].Activator();
                }
            }
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
    }
}

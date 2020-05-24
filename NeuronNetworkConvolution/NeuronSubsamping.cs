using static System.Math;

namespace NeuronNetworkConvolution
{
    class SubsampingNeuron
    {/// <summary>
     /// Конструктор нейрона
     /// </summary>
     /// <param name="inputs">Массив входных значений</param>
        public SubsampingNeuron(double[] inputs)
        {
            Inputs = inputs;
        }
        /// <summary>
        /// Массив входных значений
        /// </summary>
        public double[] Inputs { get; set; }
        /// <summary>
        /// Выходное значение
        /// </summary>
        public double Output { get; private set; }
        /// <summary>
        /// Функция активации.
        /// </summary>
        public void Activator()
        {
            Output = Max(Max(Inputs[0], Inputs[1]), Max(Inputs[2], Inputs[3]));
        }
    }
}

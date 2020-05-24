

namespace NeuronNetworkConvolution
{
    public class Convolution
    {
        /// <summary>
        /// Конструктор свертки
        /// </summary>
        /// <param name="weights">Массив весов</param>
        public Convolution(double[] weights)
        {
            Weights = weights;
        }
        /// <summary>
        /// Массив весов
        /// </summary>
        public double[] Weights { get; set; }
    }
}

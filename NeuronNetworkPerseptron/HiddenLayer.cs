
namespace NeuronNetworkPerseptron
{
    public class LayerHidden : Layer
    {
        /// <summary>
        /// Конструктор.
        /// </summary>
        /// <param name="non">Число нейронов текущего слоя.</param>
        /// <param name="nopn">Число нейронов предыдущего слоя.</param>
        /// <param name="nt">Тип нейрона.</param>
        /// <param name="type">Название слоя.</param>
        public LayerHidden(int non, int nopn, NeuronType nt, string type) : base(non, nopn, nt, type) { }
        /// <summary>
        /// Для прямых проходов.
        /// </summary>
        /// <param name="net">Сеть.</param>
        /// <param name="nextLayer">Следующий слой.</param>
        public override void Recognize(Network net, Layer nextLayer)
        {
            double[] hidden_out = new double[Neurons.Length];
            for (int i = 0; i < Neurons.Length; ++i)
                hidden_out[i] = Neurons[i].Output;
            nextLayer.Data = hidden_out;
        }
        /// <summary>
        /// Для обратных проходов.
        /// </summary>
        /// <param name="gr_sums">Массив градиентов.</param>
        /// <returns>бла</returns>
        public override double[] BackwardPass(double[] gr_sums)
        {
            double[] gr_sum = new double[numofprevneurons];
            for (int j = 0; j < gr_sum.Length; ++j)
            {
                double sum = 0;
                for (int k = 0; k < Neurons.Length; ++k)
                    sum += Neurons[k].Weights[j] * Neurons[k].Derivative * gr_sums[k];//через градиентные суммы и производную
                gr_sum[j] = sum;
            }
            for (int i = 0; i < numofneurons; ++i)
                for (int n = 0; n < numofprevneurons + 1; ++n)
                {
                    double deltaw = (n == 0) ? (momentum * lastdeltaweights[i, 0] + learningrate * Neurons[i].Derivative
                        * gr_sums[i]) : (momentum * lastdeltaweights[i, n] + learningrate * Neurons[i].Inputs[n - 1]
                        * Neurons[i].Derivative * gr_sums[i]);
                    lastdeltaweights[i, n] = deltaw;
                    Neurons[i].Weights[n] += deltaw;//коррекция весов
                }
            return gr_sum;
        }
    }
}

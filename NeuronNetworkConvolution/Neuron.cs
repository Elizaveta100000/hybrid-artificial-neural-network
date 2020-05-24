using static System.Math;

namespace NeuronNetworkConvolution
{
    public class Neuron
    {
        /// <summary>
        /// Конструктор нейрона
        /// </summary>
        /// <param name="inputs">Массив входных значений</param>
        /// <param name="weights">Массив весов</param>
        /// <param name="type">Тип нейрона</param>
        public Neuron(double[] inputs, double[] weights, NeuronType type)
        {
            this.type = type;
            Weights = weights;
            Inputs = inputs;
        }
        /// <summary>
        /// Тип нейрона
        /// </summary>
        private NeuronType type;
        /// <summary>
        /// Константы для функции активации
        /// </summary>
        private double a = 0.01d;
        /// <summary>
        /// Массив весов
        /// </summary>
        public double[] Weights { get; set; }
        /// <summary>
        /// Массив входных значений
        /// </summary>
        public double[] Inputs { get; set; }
        /// <summary>
        /// Выходное значение
        /// </summary>
        public double Output { get; private set; }
        /// <summary>
        /// Значение производной
        /// </summary>
        public double Derivative { get; private set; }
        /// <summary>
        /// Нелинейные преобразования.
        /// </summary>
        /// <param name="i">Массив входных значений.</param>
        /// <param name="w">Массив весов.</param>
        public void Activator(double[] i, double[] w)
        {
            double sum = w[0];//аффиное преобразование через смещение(нулевой вес)
            for (int l = 0; l < i.Length; ++l)
                sum += i[l] * w[l + 1];//линейные преобразования
            switch (type)
            {
                case NeuronType.Hidden:
                    Output = LeakyReLU(sum);
                    Derivative = LeakyReLU_Derivativator(sum);
                    break;
                case NeuronType.Output:
                    Output = Exp(sum);
                    break;
            }
        }
        /// <summary>
        /// Функция активации Leaky RelU.
        /// </summary>
        /// <param name="sum">Значение линейных преобразований.</param>
        /// <returns>Результат выполнения функции.</returns>
        private double LeakyReLU(double sum) => 1 / (1 + Exp(-sum));
        /// <summary>
        /// Функция производной
        /// </summary>
        /// <param name="sum">Значение линейных преобразований.</param>
        /// <returns>Результат выполнения функции.</returns>
        private double LeakyReLU_Derivativator(double sum) => Exp(-sum) / Pow((1 + Exp(-sum)), 2);
    }
}

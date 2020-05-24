using System.Collections.Generic;
using static System.Math;

namespace NeuronNetworkConvolution
{
    public class ConvolutionOfNeuron
    {
        /// <summary>
        /// Конструктор нейрона
        /// </summary>
        /// <param name="inputs">Массив входных значений</param>
        /// <param name="convolution">Свертка</param>
        public ConvolutionOfNeuron(double[] inputs, Convolution convolution)
        {
            Conv = convolution;
            Inputs = inputs;
            numbInp = new List<int>();
        }
        /// <summary>
        /// Свертка
        /// </summary>
        public Convolution Conv { get; set; }
        /// <summary>
        /// Массив входных значений
        /// </summary>
        public double[] Inputs { get; set; }
        /// <summary>
        /// Константы для функции активации
        /// </summary>
        private double a = 0.01d;
        /// <summary>
        /// Выходное значение
        /// </summary>
        public double Output { get; private set; }
        /// <summary>
        /// Значение производной
        /// </summary>
        public double Derivative { get; private set; }
        /// <summary>
        /// Номера входных пикселей.
        /// </summary>
        public List<int> numbInp { get; set; }
        /// <summary>
        /// Функция активации.
        /// </summary>
        public void Activator()
        {
            double sum = 0;
            for (int l = 0; l < Inputs.Length; ++l)
            {
                sum += Inputs[l] * Conv.Weights[l]; // линейные преобразования
            }
            Output = LeakyReLU(sum);
            Derivative = LeakyReLU_Derivativator(sum);
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

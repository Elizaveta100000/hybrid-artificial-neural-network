
namespace NeuronNetworkConvolution
{
    /// <summary>
    /// Режимы работы памяти (GET, SET)
    /// </summary>
    public enum MemoryMode
    {
        /// <summary>
        /// Запись.
        /// </summary>
        GET,
        /// <summary>
        /// Чтение.
        /// </summary>
        SET
    }
    /// <summary>
    /// Тип нейрона (Hidden, Output)
    /// </summary>
    public enum NeuronType
    {
        /// <summary>
        /// Скрытый слой.
        /// </summary>
        Hidden,
        /// <summary>
        /// Выходной слой.
        /// </summary>
        Output
    }
    /// <summary>
    /// Режимы работы сети (Train, Test, Demo)
    /// </summary>
    public enum NetworkMode
    {
        /// <summary>
        /// Обучение.
        /// </summary>
        Train,
        /// <summary>
        /// Тестирование.
        /// </summary>
        Test,
        /// <summary>
        /// Демонстрация.
        /// </summary>
        Demo
    }
}

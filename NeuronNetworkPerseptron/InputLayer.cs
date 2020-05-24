using System.Drawing;
using System.IO;
using ImageEXcs;
using System;

namespace NeuronNetworkPerseptron
{
    public class LayerInput
    {
        public LayerInput() { }
        /// <summary>
        /// Конструктор.
        /// </summary>
        /// <param name="nm">Режим работы сети.</param>
        public LayerInput(NetworkMode nm)
        {
            Image img;
            int[,] imgByte;
            double[] imgVect;
            //System.Drawing.Bitmap bitmap;
            switch (nm)
            {
                case NetworkMode.Train:
                    string[] fullfilesPath = Directory.GetFiles(@"TrainingSample\", "*.*", SearchOption.AllDirectories);
                    int len = fullfilesPath.Length;
                    Trainset = new(double[], byte)[len];
                    for (int numbImg = 0; numbImg < len; numbImg++)
                    {
                        img = Image.FromFile(fullfilesPath[numbImg]);
                        //ImageEXcs.ImageEXcs.ToInputIm(img, 32, 32).Save($"{numbImg}.jpg");
                        imgByte = ImageEXcs.ImageEXcs.ToInput(img, 32, 32);
                        imgVect = new double[32 * 32];
                        for (int i = 0; i < imgVect.Length; i++)
                        {
                            int row = (int)i / 32;
                            imgVect[i] = imgByte[row, i - row * 32];
                        }
                        Trainset[numbImg].Item1 = imgVect;
                        string letter = Path.GetFileName(fullfilesPath[numbImg]);
                        Trainset[numbImg].Item2 = symb(letter);
                    }
                    break;
                case NetworkMode.Test:
                    string[] testfilesPath = Directory.GetFiles(@"TestingSample\", "*.*", SearchOption.AllDirectories);
                    int lenTest = testfilesPath.Length;
                    Testset = new(double[], byte)[lenTest];
                    for (int numbImg = 0; numbImg < lenTest; numbImg++)
                    {
                        img = Image.FromFile(testfilesPath[numbImg]);
                        //ImageEXcs.ImageEXcs.ToInputIm(img, 32, 32).Save($"{numbImg}.jpg");
                        imgByte = ImageEXcs.ImageEXcs.ToInput(img, 32, 32);
                        imgVect = new double[32 * 32];
                        for (int i = 0; i < imgVect.Length; i++)
                        {
                            int row = (int)i / 32;
                            imgVect[i] = imgByte[row, i - row * 32];
                        }
                        Testset[numbImg].Item1 = imgVect;
                        string letter = Path.GetFileName(testfilesPath[numbImg]);
                        Testset[numbImg].Item2 = symb(letter);
                    }
                    break;
            }
        }

        private byte symb(string letter)
        {
            byte bt = 0;
            switch (letter)
            {
                case "а.PNG":
                    bt = 0;
                    break;
                case "б.PNG":
                    bt = 1;
                    break;
         
                default:
                    return 1;
                    break;
            }
            return bt;
        }

        public double[] output(LayerInput inp, int j)
        {
            double[] fc = new double[2];
            double min = 0.003;
            double max = 0.012;
            double mi = 0.9;
            double ma = 0.97;
            j = (int)inp.Testset[j].Item2;
            if (j != 34)
                for (int k = 0; k < 2; k++)
                    fc[k] = k == j ? random.NextDouble() * (ma - mi) + mi : random.NextDouble() * (max - min) + min;
            else
                return null;
            return fc;
        }

        private System.Random random = new System.Random();
        /// <summary>
        /// Обучающая выборка.
        /// </summary>
        public (double[], byte)[] Trainset { get; }
        /// <summary>
        /// Тестирующая выборка.
        /// </summary>
        public (double[], byte)[] Testset { get; }
    }
}

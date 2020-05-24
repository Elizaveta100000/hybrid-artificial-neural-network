using System.Drawing;

namespace ImageEXcs
{
    public static class ImageEXcs
    {
        /// <summary>
        /// Максимальный цвет RGB для символа
        /// </summary>
        private static int col = 110;
        /// <summary>
        /// Перевод картинки в массив 0 и 1
        /// </summary>
        /// <param name="img">Картинка</param>
        /// <returns>Массив картинки в 0 и 1</returns>
        public static int[,] ToByte(this Image img)
        {
            var bmp = new Bitmap(img);
            int[,] mass = new int[bmp.Width, bmp.Height];

            for (int y = 0; y < img.Height; y++)
            {
                for (int x = 0; x < img.Width; x++)
                {
                    var IsWhite = bmp.GetPixel(x, y).R >= col && bmp.GetPixel(x, y).G >= col && bmp.GetPixel(x, y).B >= col;
                    mass[x, y] = IsWhite ? 0 : 1;
                }
            }
            return mass;
        }
        /// <summary>
        /// Перевод массива 0 и 1 в ч/б картинку
        /// </summary>
        /// <param name="img">Массив картинки в 0 и 1</param>
        /// <returns>Картинка</returns>
        public static Image ToImage(this int[,] img)
        {
            var bmp = new Bitmap(img.GetLength(0), img.GetLength(1));

            for (int y = 0; y < bmp.Height; y++)
                for (int x = 0; x < bmp.Width; x++)
                {
                    bmp.SetPixel(x, y, img[x, y] == 1 ? Color.Black : Color.White);
                }
            return (Image)bmp;
        }
        /// <summary>
        /// Удаление лишних краев
        /// </summary>
        /// <param name="bytes">Массив картинки в 0 и 1</param>
        /// <returns>Массив картинки в 0 и 1</returns>
        public static int[,] CutNumber(this int[,] bytes)
        {
            var r = getRect(bytes);

            var mass = new int[bytes.GetLength(0) - r.X, bytes.GetLength(1) - r.Y];
            for (int y = 0; y < mass.GetLength(1); y++)
                for (int x = 0; x < mass.GetLength(0); x++)
                {
                    mass[x, y] = bytes[x + r.X, y + r.Y];
                }
            return mass;
        }
        /// <summary>
        /// Изменение размера картинки
        /// </summary>
        /// <param name="source">Картинка</param>
        /// <param name="width">Ширина</param>
        /// <param name="height">Высота</param>
        /// <returns>Картинка</returns>
        public static Image ScaleImage(this Image source, int width, int height)
        {
            Bitmap bmp = new Bitmap(source, width, height);
            return (Image)bmp;
        }
        /// <summary>
        /// Перевод картинки в массив 0 и 1, изменив размер
        /// </summary>
        /// <param name="source">Картинка</param>
        /// <param name="height">Ширина</param>
        /// <param name="width">Высота</param>
        /// <returns>Массив картинки в 0 и 1</returns>
        public static int[,] ToInput(this Image source, int width, int height)
        {
            return source.ToByte().CutNumber().ToImage().ScaleImage(width, height).ToByte();
        }
        /// <summary>
        /// Очищение картинки от мусора и перевод в картинку заданного размера
        /// </summary>
        /// <param name="source">Картинка</param>
        /// <param name="height">Ширина</param>
        /// <param name="width">Высота</param>
        /// <returns>Картинка</returns>
        public static Image ToInputIm(this Image source, int width, int height)
        {
            return source.ToInput(width, height).ToImage();
        }

        /// <summary>
        /// Определение краев картинки
        /// </summary>
        /// <param name="bytes">Массив картинки в 0 и 1</param>
        /// <returns>Координаты обрезки</returns>
        public static Point getRect(int[,] bytes)
        {
            Point pt = new Point();
            for (int x = 0; x < bytes.GetLength(0); x++)
            {
                bool flag = false;
                int k = 0;
                for (int y = 0; y < bytes.GetLength(1); y++)
                {
                    if (bytes[x, y] == 1)
                    {
                        k++;
                        if (k > 1)
                        {
                            pt.X = x;
                            flag = true;
                            break;
                        }
                    }
                }
                if (flag)
                    break;
            }
            for (int y = 0; y < bytes.GetLength(1); y++)
            {
                bool flag = false;
                int k = 0;
                for (int x = 0; x < bytes.GetLength(0); x++)
                {
                    if (bytes[x, y] == 1)
                    {
                        k++;
                        if (k > 1)
                        {
                            pt.Y = y;
                            flag = true;
                            break;
                        }
                    }
                }
                if (flag)
                    break;
            }
            return pt;
        }
    }
}

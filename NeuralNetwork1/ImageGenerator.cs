﻿using System;
using System.Drawing;

namespace NeuralNetwork
{
    /// <summary>
    /// Тип фигуры
    /// </summary>
    public enum FigureType : byte { Triangle = 0, Rectangle, Circle, Sinusiod, Undef };
    
    public class GenerateImage
    {
        /// <summary>
        /// Бинарное представление образа
        /// </summary>
        public bool[,] img = new bool[200, 200];
        
        //  private int margin = 50;
        private readonly Random _rand = new Random();
        
        /// <summary>
        /// Текущая сгенерированная фигура
        /// </summary>
        public FigureType currentFigure = FigureType.Undef;

        /// <summary>
        /// Количество классов генерируемых фигур (4 - максимум)
        /// </summary>
        public int FigureCount { get; set; } = 4;

        /// <summary>
        /// Диапазон смещения центра фигуры (по умолчанию +/- 20 пикселов от центра)
        /// </summary>
        public int FigureCenterGitter { get; set; } = 20;

        /// <summary>
        /// Диапазон разброса размера фигур
        /// </summary>
        public int FigureSizeGitter { get; set; } = 20;

        /// <summary>
        /// Диапазон разброса размера фигур
        /// </summary>
        public int FigureSize { get; set; } = 100;
        
        /// <summary>
        /// Очистка образа
        /// </summary>
        public void ClearImage()
        {
            for (int i = 0; i < 200; ++i)
                for (int j = 0; j < 200; ++j)
                    img[i, j] = false;
        }

        public Sample GenerateFigure()
        {
            generate_figure();
            double[] input = new double[400];
            for (int i = 0; i < 400; i++)
                input[i] = 0;

            FigureType type = currentFigure;

            for (int i = 0; i < 200; i++)
                for (int j = 0; j < 200; j++)
                    if (img[i, j])
                    { 
                        input[i] += 1;
                        input[200 + j] += 1;
                    }
            return new Sample(input, type);
        }

        private Point GetLeftUpperPoint()
        {
            int X = 100 - FigureSize / 2 + _rand.Next(-FigureSizeGitter / 2, FigureSizeGitter / 2);
            int Y = 100 - FigureSize / 2 + _rand.Next(-FigureSizeGitter / 2, FigureSizeGitter / 2);
            return new Point(X,Y);
        }

        private Point GetRightDownPoint()
        {
            int X = 100 + FigureSize / 2 + _rand.Next(-FigureSizeGitter / 2, FigureSizeGitter / 2);
            int Y = 100 + FigureSize / 2 + _rand.Next(-FigureSizeGitter / 2, FigureSizeGitter / 2);
            return new Point(X, Y);
        }

        private Point GetCenterPoint()
        {
            int X = 100 + _rand.Next(-FigureSizeGitter / 2, FigureSizeGitter / 2);
            int Y = 100 + _rand.Next(-FigureSizeGitter / 2, FigureSizeGitter / 2);
            return new Point(X, Y);
        }


        private void Bresenham(int x, int y, int x2, int y2)
        {
            int w = x2 - x;
            int h = y2 - y;
            int dx1 = 0, dy1 = 0, dx2 = 0, dy2 = 0;
            if (w < 0) dx1 = -1; else if (w > 0) dx1 = 1;
            if (h < 0) dy1 = -1; else if (h > 0) dy1 = 1;
            if (w < 0) dx2 = -1; else if (w > 0) dx2 = 1;
            int longest = Math.Abs(w);
            int shortest = Math.Abs(h);

            if (!(longest > shortest))
            {
                longest = Math.Abs(h);
                shortest = Math.Abs(w);
                if (h < 0) dy2 = -1; else if (h > 0) dy2 = 1;
                dx2 = 0;
            }

            int numerator = longest >> 1;
            for (int i = 0; i <= longest; i++)
            {
                img[x, y] = true;
                numerator += shortest;
                if (!(numerator < longest))
                {
                    numerator -= longest;
                    x += dx1;
                    y += dy1;
                }
                else
                {
                    x += dx2;
                    y += dy2;
                }
            }
        }

        public bool create_triangle()
        {
            currentFigure = FigureType.Triangle;
            Point leftUpper = GetLeftUpperPoint();
            Point downLeft = GetRightDownPoint();
            int centerX = 100 + FigureCenterGitter;


            Bresenham(leftUpper.X, downLeft.Y, centerX, leftUpper.Y);
            Bresenham(centerX, leftUpper.Y, downLeft.X, downLeft.Y);
            Bresenham(downLeft.X, downLeft.Y, leftUpper.X, downLeft.Y);

            return true;
        }

        public bool create_rectangle()
        {
            currentFigure = FigureType.Rectangle;

            Point leftUpper = GetLeftUpperPoint();
            Point downLeft = GetRightDownPoint();

            Bresenham(leftUpper.X, leftUpper.Y, downLeft.X, leftUpper.Y);
            Bresenham(downLeft.X, leftUpper.Y, downLeft.X, downLeft.Y);
            Bresenham(downLeft.X, downLeft.Y, leftUpper.X, downLeft.Y);
            Bresenham(leftUpper.X, downLeft.Y, leftUpper.X, leftUpper.Y);
            return true;
        }

        public bool create_circle()
        {
            currentFigure = FigureType.Circle;

            Point center = GetCenterPoint();

            int radius = _rand.Next(50, 65);

            for (double t = 0; t < 2 * Math.PI; t += 0.01)
            {
                double x = center.X + radius * Math.Cos(t);
                double y = center.Y + radius * Math.Sin(t);
                img[(int)x, (int)y] = true;
            }
            return true;
        }

        public bool create_sin()
        {
            currentFigure = FigureType.Sinusiod;

            Point leftUpper = GetLeftUpperPoint();
            Point downLeft = GetRightDownPoint();

            int amp = (downLeft.Y - leftUpper.Y) / 2;
            int centerY = leftUpper.Y + amp;

            double sx = 0.25;
            for (double x = leftUpper.X; x <= downLeft.X; x += 0.05)
            {
                double y = Math.Round(centerY + amp * Math.Sin(sx * x));
                img[(int)x, (int)y] = true;
            }

            return true;
        }


        public void generate_figure(FigureType type = FigureType.Undef)
        {

            if (type == FigureType.Undef || (int)type >= FigureCount)
                type = (FigureType)_rand.Next(FigureCount);
            ClearImage();
            switch (type)
            {
                case FigureType.Rectangle : create_rectangle(); break;
                case FigureType.Triangle  : create_triangle(); break;
                case FigureType.Circle    : create_circle(); break;
                case FigureType.Sinusiod  : create_sin(); break;

                default:
                    type = FigureType.Undef;
                    throw new Exception("Ошибка создания фигуры!");
            }
        }

        /// <summary>
        /// Возвращает битовое изображение для вывода образа
        /// </summary>
        /// <returns></returns>
        public Bitmap GenBitmap()
        {
            Bitmap DrawArea = new Bitmap(200, 200);
            for (int i = 0; i < 200; ++i)
                for (int j = 0; j < 200; ++j)
                    if (img[i, j])
                        DrawArea.SetPixel(i, j, Color.Black);
            return DrawArea;
        }
    }

}

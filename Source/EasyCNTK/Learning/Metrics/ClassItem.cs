//
// Copyright (c) Stanislav Grigoriev. All rights reserved.
// grigorievstas9@gmail.com 
// https://github.com/StanislavGrigoriev/EasyCNTK
//
// Licensed under the MIT license. See LICENSE.txt file in the project root for full license information.
//

namespace EasyCNTK.Learning.Metrics
{
    /// <summary>
    /// Содержит дополнительные метрики классификации по конкретному классу
    /// </summary>
    public class ClassItem
    {
        /// <summary>
        /// Индекс позиции в выходном векторе модели, закрепленной за определенным классом
        ///</summary>
        public int Index { get; set; }
        /// <summary>
        /// Точность, с которой модель определяет этот класс. Вычисляется по формуле: точность = [количество верно определенных примеров этого класса] / [количество примеров классифицированных как этот класс]
        /// </summary>
        public double Precision { get; set; }
        /// <summary>
        /// Полнота, с которой модель определяет этот класс. Вычисляется по формуле: полнота = [количество верно определенных примеров этого класса] / [количество всех примеров этого класса]
        /// </summary>
        public double Recall { get; set; }
        /// <summary>
        /// Средняя гармоника между <seealso cref="Precision"/> и <seealso cref="Recall"/>. Вычисляется по формуле: F1Score = 2 * Precision * Recall / (Precision + Recall)
        /// </summary>
        public double F1Score { get; set; }
        /// <summary>
        /// Доля примеров данного класса во всем датасете
        /// </summary>
        public double Fraction { get; set; }        
    }
}

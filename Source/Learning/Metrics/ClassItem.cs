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
    /// Содержит дополнительную информацию о классификации по конкретному классу
    /// </summary>
    public class ClassItem
    {
        /// <summary>
        /// Индекс позиции в выходном векторе модели, закрепленной за определенным классом
        ///</summary>
        public int Index { get; set; }
        /// <summary>
        /// Точность, с которой модель определяет этот класс. Вычисляется по формуле: точность = [количество верно определенных примеров этого класса] / [количество всех примеров этого класса]
        /// </summary>
        public double Precision { get; set; }
        /// <summary>
        /// Доля примеров данного класса во всем датасете
        /// </summary>
        public double Fraction { get; set; }        
    }
}

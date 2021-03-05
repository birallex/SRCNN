# SRCNN
This is super resolution convolutional neural network for image restoration or for the other wide-range tasks


Данная нейросеть подходит для широкого спектра задач по коррекции изобрежений.
В перспективе можно "научить" её не только улучшать изображение, но и изменять его согласно определённым критериям. 

Пример входного изображения:

<p align="center">
  <img src="https://github.com/birallex/SRCNN/blob/main/weights/input_example.jpg" width="528" height="294"/>
</p>

Тот же фрагмент изображения, прошедший через нейросеть с обученными весами на малом датасете(фото баркодов):

<p align="center">
  <img src="https://github.com/birallex/SRCNN/blob/main/weights/output_example_srcnn.jpg" width="528" height="294"/>
</p>


При необходимости переобучить нейросеть на другую задачу, необходимо загрузить свои фото в следующие папки:

    1) dataset/photos - изображения для обучения
    
    2) dataset/valid_photos - изображения для валидации сети во время обучения

Далее необходимо запустить скрипт make_dataset.py для создания датасета для обучения сети. Файлы появятся в директории и делать с ними ничего не надо.

Далее запускаем скрипт train.py и после получения новых весов можем использовать нейросеть.

После проведения вышеуказанных операций можно загружать свои изображения в директорию input/
 и запуcтить process.py. 
 
 По его завершению вы получите обработанные изображения в директории output/
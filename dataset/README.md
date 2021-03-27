# Создание датасета

Для создания ремурсов для обучения и валидации необходимо запустить скрипт `make_dataset.py`.
Как пример: `python3 make_dataset.py`. По завершению работы скрипта будут получены два файла с расширением `.h5`,
 после чего можно приступать к обучению.

To create training dataset use `make_dataset.py` script: `python3 make_dataset.py`. After that you get two files with `.h5` extention. All done :) 

# Уменьшение размера тренировочных данных

При слишком большом разрешении изображений может просто не хватить ОЗУ. В таком случае необходимо уменьшить разрешение
 изображений, воспользовавшись одной из функций скрипта `resizer.py`.

### Варианты таковы: 
- `process_folder_current_resolution()` приведёт все изображения к заданному разрешению
- `process_folder()` приведёт изображения к разрешению согласно коэффициенту масштабирования 

If the size of the dataset is too large or resolutions of the images too big use `resizer.py` functions to reduce image resolution.
Reducing the resolution directly affects the size of the output files. 
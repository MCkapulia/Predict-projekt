1) Роли участников:
1. Капитунова,Уланова - создание модели и расчет прогнозов
2. Муратова,Федосеева - получение данных с тинькофф-инвестиций для прогноза и текущей стоимости акции
3. Благодырь,Русакова - создание телеграм бота 

2) Структура:

Приложение для предсказания цен акций Роснефти состоит из следующих файлов, каждый из которых реализует определенную функциональность: 

Пользователи:
- Вводят команды в телеграм-бота для получения прогнозов цен акций Роснефти. 

Телеграм-бот:
- Получает команды от пользователей. 
- Передает команды файлу telegramBot.py для дальнейшей обработки. 

Файл telegramBot.py:
- Принимает команды от телеграм-бота.
- Обрабатывает команды, запрашивая данные у tinkoffPrice.py. 
- Отправляет запросы на предсказание в ModelCreate.py. 
- Выводит результаты пользователям. 

Файл tinkoffPrice.py:
- Получает данные о текущей цене акций. 
- Получает данные за последние 100 дней для использования в прогнозировании. 
- Передает данные telegramBot.py для отображения текущего курса и использования в прогнозировании. 
- Сохраняет данные в файлы Excel с помощью pandas для дальнейшего использования в обучении модели. 

Файл ModelCreate.py:
- Обучает модель на исторических данных. 
- Использует библиотеку keras для создания и обучения модели LSTM. 
- Считывает параметры нормализации данных. 
- Получает данные от telegramBot.py и возвращает предсказания.

Взаимодействия компонентов: 
- Пользователи вводят команды в телеграм-бота.
- Телеграм-бот передает команды файлу telegramBot.py.

Файл telegramBot.py:
- Запрашивает текущие данные о ценах акций у файла tinkoffPrice.py. 
- Получает данные о ценах за последние 100 дней от файла tinkoffPrice.py. 
- Передает данные для предсказания в файл ModelCreate.py. 
- Получает предсказания от файла ModelCreate.py. 
- Выводит текущие данные и предсказания пользователям. 

Файл tinkoffPrice.py:
- Запрашивает данные о текущей цене акций. 
- Запрашивает исторические данные (цены за последние 100 дней). 
- Сохраняет данные в файлы Excel с помощью pandas. 
- Передает текущие и исторические данные файлу telegramBot.py для использования в прогнозировании. 

Файл ModelCreate.py:
- Обучает модель LSTM на исторических данных. 
- Получает данные от telegramBot.py для предсказания. 
- Возвращает предсказания файлу telegramBot.py.


3) Сборка:
1. скачать с gitHub проект
2. зарегистрироваться на тинькофф-инвестиции
3. получить токен в тинькофф-инвестициях
4. найти в телеграме бота @BotFather и отправить ему команду "newbot"
5. после создания своего бота, вы получите токен
6. открыть проект, создать в нем файлы с расширением .env и записать в него два токена
7. установить соответствующие библиотеки
8. бот готов к использованию!

4) Запуск:
1. пользователю необходимо отсканировать QR-код,далее его перенаправят в чат телеграм.
2. пользователь отправляет боту команду /start.
3. перед пользователем появляется 3 кнопки ( 1 кнопка - прогноз; 2 кнопка - текущий курс; 3 кнопка - переобучение модели).
4. пользователь делает необходимый запрос.

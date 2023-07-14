# RL-based Recommender training in synthetic environment


Рекомендательные системы помогают пользователям сократить время на поиск нужных товаров или контента, 
а также найти что-то незнакомое, но интересное, например, нового исполнителя или сериал. 
Хорошая рекомендательная система исследует и выявляет интересы пользователя, 
демонстрируя перспективные рекомендации и получая от пользователя отклики, 
например, клики, лайки, покупки. 

Обучение рекомендательной системы с элементами exploration в реальной среде 
(на откликах на рекомендации от реальных пользователей) может привести к негативному 
пользовательскому опыту и занять много времени. 
Альтернативный способ - моделирование откликов пользователей с использованием симулятора 
для рекомендательных систем для обучения online RL-моделей и оценки качества рекомендательных систем. 

Sber AI Lab совместно с университетом ИТМО разработал симулятор для рекомендательных систем
[Sim4Rec](https://github.com/sb-ai-lab/Sim4Rec), 
которым мы предлагаем воспользоваться как средой для обучения моделей. 
 
Мы предлагаем участникам проекта исследовать различные подходы к построению RL-based рекомендаций
и оценить их performance, преимущества и недостатки для различных моделей отклика.

### Модели отклика

Мы разработали следующие модели отклика, для которых необходимо построить рекомендательную модель:
- [task 1](./task_1.ipynb) Есть несколько популярных товаров, которые пользователи выбирают чаще всего 
- [task 2](./task_2.ipynb) У пользователей есть социально-демографические признаки, а у товаров - характеристики. 
Вероятность отклика пользователей определяется моделью, обученной на данных о предыдущих взаимодействиях пользователей и товаров.
- [task 3](./task_3.ipynb) Продолжение task_1, теперь популярность товаров меняется со временем, и хорошая модель будет учитывать эти изменения
- [task 4](/task_4.ipynb) Бонусная модель отклика для самых энергичных и умелых! У пользователей есть персональные предпочтения, 
определенные на базе из прошлой истории взаимодействия с товарами. Вам доступны только id.


### Рекомендательные модели
В заданиях 1 и 3 вам пригодятся неперсонализированные модели. 
Начните с реализации и сравнения [multi-armed bandits для задачи рекомендаций](https://eugeneyan.com/writing/bandits/). 

В задании 2 необходимо учесть признаки, что достичь лучшего качества, чем возможно с multi-armed bandits. 
Начните с [LinUCB](https://arxiv.org/pdf/1003.0146.pdf).

В задании 4 вы можете комбинировать [стандартные рекомендательные модели, такие как ALS, itemKNN.](https://sb-ai-lab.github.io/RePlay/pages/modules/models.html) 
с exploration или использовать различные доступные RL-based подходы к рекомендациям, такие как [DDPG](https://dl.acm.org/doi/abs/10.1145/3523227.3551485).

### Полезные ссылки
[Обзор симуляторов для рекомендательных систем](https://arxiv.org/pdf/2206.11338.pdf)
[Симулятор для рекомендательных систем Sim4Rec](https://github.com/sb-ai-lab/Sim4Rec)
[Библиотека алгоритмов для рекомендательных систем RePlay](https://github.com/sb-ai-lab/RePlay/)

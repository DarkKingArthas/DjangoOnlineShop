import pandas as pd
from shop.models import Myrating


# Получение всех записей из модели Myrating
ratings = Myrating.objects.all()

# Создание пустого списка для данных
data = []

# Итерация по объектам Myrating и добавление данных в список
for rating in ratings:
    rating_data = {
        'user_id': rating.user_id,
        'product_id': rating.product_id,
        'rating': rating.rating
    }
    data.append(rating_data)

# Создание датафрейма из списка данных
df = pd.DataFrame(data)

# Вывод датафрейма для проверки
print(df)
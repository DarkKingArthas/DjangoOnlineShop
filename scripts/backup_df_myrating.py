import os
import pandas as pd
import django
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'bookstore.settings')
django.setup()

from shop.models import Myrating, Product

# Получение всех записей из модели Myrating
ratings = Myrating.objects.all()
products = Product.objects.all()

# Создание пустого списка для данных
data = []

# Итерация по объектам Myrating и добавление данных в список
for rating in ratings:
    rating_data = {
        'user_id': rating.user_id,
        'product_id': rating.product_id,
        'rating': rating.rating,
    }
    data.append(rating_data)

# Создание датафрейма из списка данных
df = pd.DataFrame(data)

# Создание объекта Reader для указания диапазона значений рейтингов
reader = Reader(rating_scale=(1, 5))

# Создание объекта Dataset из данных
data = Dataset.load_from_df(df[['user_id', 'product_id', 'rating']], reader)

# Разделение данных на обучающий и тестовый наборы
trainset, testset = train_test_split(data, test_size=0.2)

# Создание модели SVD
model = SVD()

# Обучение модели на обучающем наборе данных
model.fit(trainset)

# Пример генерации рекомендаций для пользователя с user_id=1
user_id = 3

# Получение списка всех продуктов
all_products = [product.id for product in products]

# Получение списка продуктов, которые пользователь с user_id=1 еще не оценил
unrated_products = [product for product in all_products if product not in df[df['user_id'] == user_id]['product_id'].values]

# Генерация рекомендаций
recommendations = []
for product_id in unrated_products:
    predicted_rating = model.predict(user_id, product_id).est
    recommendations.append({
        'product_id': product_id,
        'predicted_rating': predicted_rating
    })

# Преобразование рекомендаций в датафрейм
recommendations_df = pd.DataFrame(recommendations)

# Сортировка рекомендаций по убыванию предсказанного рейтинга
if not recommendations_df.empty:
    recommendations_df = recommendations_df.sort_values('predicted_rating', ascending=False)
    # Вывод рекомендации
    for index, row in recommendations_df.iterrows():
        product_id = row['product_id']
        predicted_rating = row['predicted_rating']
        product_title = Product.objects.get(id=product_id).name
        print(f"Recommendation {index + 1}: Product '{product_title}', predicted rating: {predicted_rating}")
else:
    print("Вы оценили все возможные книги. В данный момент для вас нет рекомендаций")

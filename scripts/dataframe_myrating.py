import os
import pandas as pd
import django
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'bookstore.settings')
django.setup()

from shop.models import Myrating, Product

ratings = Myrating.objects.all()
products = Product.objects.all()

data = []

for rating in ratings:
    rating_data = {
        'user_id': rating.user_id,
        'product_id': rating.product_id,
        'rating': rating.rating,
    }
    data.append(rating_data)

df = pd.DataFrame(data)

# Добавление столбцов 'author' и 'publisher' в датафрейм
df['author'] = df['product_id'].apply(lambda x: Product.objects.get(id=x).author)
df['publisher'] = df['product_id'].apply(lambda x: Product.objects.get(id=x).publisher)

# Преобразование текстовых значений автора и издательства в числовые коды
author_encoder = LabelEncoder()
publisher_encoder = LabelEncoder()
df['author_code'] = author_encoder.fit_transform(df['author'])
df['publisher_code'] = publisher_encoder.fit_transform(df['publisher'])

reader = Reader(rating_scale=(1, 5))

data = Dataset.load_from_df(df[['user_id', 'product_id', 'rating']], reader)

trainset, testset = train_test_split(data, test_size=0.2)

model = SVD()

model.fit(trainset)

user_id = 3

all_products = [product.id for product in products]

unrated_products = [product for product in all_products if product not in df[df['user_id'] == user_id]['product_id'].values]

recommendations = []
for product_id in unrated_products:
    predicted_rating = model.predict(user_id, product_id).est
    author = Product.objects.get(id=product_id).author
    publisher = Product.objects.get(id=product_id).publisher
    recommendations.append({
        'product_id': product_id,
        'predicted_rating': predicted_rating,
        'author': author,
        'publisher': publisher,
    })

recommendations_df = pd.DataFrame(recommendations)
print(df)

if not recommendations_df.empty:
    recommendations_df = recommendations_df.sort_values('predicted_rating', ascending=False)
    for index, row in recommendations_df.iterrows():
        product_id = row['product_id']
        predicted_rating = row['predicted_rating']
        product_title = Product.objects.get(id=product_id).name
        author = row['author']
        publisher = row['publisher']
        print(f"Рекоммендация {index + 1}: Продукт '{product_title}', Автор: {author}, Издательство: {publisher}, Предсказанный рейтинг: {predicted_rating}")
else:
    print("Вы оценили все возможные книги. Для вас в данный момент рекомендаций нет")

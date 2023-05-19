from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth import authenticate, login, logout
from django.http import HttpResponseRedirect
from django.http import Http404
from django.db.models import Q
from django.contrib.auth.models import User
from django.urls import reverse
from sklearn.preprocessing import LabelEncoder

from .models import Category, Product, Myrating
from django.contrib import messages
from cart.forms import CartAddProductForm
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from django.db.models import Case, When
from .recommendation import Myrecommend
import numpy as np
import pandas as pd


# for recommendation
def recommend(request):
    if not request.user.is_authenticated:
        return redirect("login")
    if not request.user.is_active:
        raise Http404

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

    current_user_id = request.user.id

    all_products = [product.id for product in products]

    unrated_products = [product for product in all_products if product not in df[df['user_id'] == current_user_id]['product_id'].values]

    recommendations = []
    for product_id in unrated_products:
        predicted_rating = model.predict(current_user_id, product_id).est
        author = Product.objects.get(id=product_id).author
        publisher = Product.objects.get(id=product_id).publisher
        recommendations.append({
            'product_id': product_id,
            'predicted_rating': predicted_rating,
            'author': author,
            'publisher': publisher,
        })

    recommendations_df = pd.DataFrame(recommendations)

    if not recommendations_df.empty:
        recommendations_df = recommendations_df.sort_values('predicted_rating', ascending=False)
        product_list = []
        for index, row in recommendations_df.iterrows():
            product_id = row['product_id']
            predicted_rating = row['predicted_rating']
            product = Product.objects.get(id=product_id)
            product_dict = {
                'get_absolute_url': product.get_absolute_url,
                'image': product.image,
                'name': product.name,
                'category': product.category,
                'price': product.price,
                'stock': product.stock,
                'predicted_rating': predicted_rating,
            }
            product_list.append(product_dict)
        return render(request, 'shop/recommend.html', {'product_list': product_list})
    else:
        message = "Вы оценили все возможные книги. Нет рекомендаций."
        return render(request, 'shop/recommend.html', {'message': message})


# List
def product_list(request, category_slug=None):
    category = None
    categories = Category.objects.all()
    products = Product.objects.filter(available=True)
    search_term = ''
    if category_slug:
        category = get_object_or_404(Category, slug=category_slug)
        products = Product.objects.filter(category=category)

    if 'search' in request.GET:
        search_term = request.GET['search']
        if search_term:
            products = Product.objects.filter(name__icontains=search_term)
            if products:
                messages.success(request, 'Найдены результаты по запросу: ' + search_term)
            else:
                messages.success(request, 'Ничего не найдено :(')
        else:
            messages.warning(request, 'Ваш запрос пуст')

    query = request.GET.get('q')
    if query:
        products = Product.objects.filter(Q(title__icontains=query)).distinct()
        return render(request, 'shop/list.html', {'products': products})

    context = {
        'category': category,
        'categories': categories,
        'products': products,
        'search_term': search_term
    }
    return render(request, 'shop/list.html', context)


# detail
def product_detail(request, id, slug):
    if not request.user.is_authenticated:
        return redirect("login")
    if not request.user.is_active:
        raise Http404
    product = get_object_or_404(Product, id=id, slug=slug, available=True)

    cart_product_form = CartAddProductForm()

    # rating
    if request.method == "POST":
        rate = request.POST['rating']
        ratingObject = Myrating()
        ratingObject.user = request.user
        ratingObject.product = product
        ratingObject.rating = rate
        ratingObject.save()
        messages.success(request, "Ваш отзыв добавлен ")
        return redirect('shop:product_list')

    context = {
        'product': product,
        'cart_product_form': cart_product_form,
    }

    return render(request, 'shop/detail.html', context)

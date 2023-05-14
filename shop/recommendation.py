import numpy as np
import pandas as pd
from .models import Myrating
from .models import Product
from .models import User

import scipy.optimize


def Myrecommend():
    ratings = Myrating.objects.all()
    books = Product.objects.all()
    num_users = User.objects.count()
    num_books = books.count()

    book_ids = [book.id for book in books]
    user_ids = [user.id for user in User.objects.all()]

    prediction_matrix = np.zeros((num_books, num_users))

    for rating in ratings:
        book_index = book_ids.index(rating.product_id)
        user_index = user_ids.index(rating.user_id)
        prediction_matrix[book_index, user_index] = rating.rating

    Ymean = np.mean(prediction_matrix, axis=1)
    Ymean = Ymean.reshape((num_books, 1))

    R = np.isnan(prediction_matrix)
    prediction_matrix[R] = 0

    X = np.random.rand(num_books, 10)
    Theta = np.random.rand(num_users, 10)

    num_iterations = 100
    learning_rate = 0.001
    reg_lambda = 0.01

    for i in range(num_iterations):
        error = np.dot(X, Theta.T) - prediction_matrix
        error[R] = 0

        X_grad = np.dot(error, Theta) + reg_lambda * X
        Theta_grad = np.dot(error.T, X) + reg_lambda * Theta

        X -= learning_rate * X_grad
        Theta -= learning_rate * Theta_grad

    prediction_matrix = np.dot(X, Theta.T) + Ymean

    return prediction_matrix, Ymean

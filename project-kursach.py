from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.offline import plot
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

# Загрузка данных
data = pd.read_csv('online_shoppers_intention.csv')

# Подготовка данных для моделей машинного обучения
X = data[['BounceRates', 'ExitRates']]  # Выбираем признаки для обучения
y = data['Revenue']  # Целевая переменная

# Разделение данных на обучающий и тестовый наборы
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Обучение моделей машинного обучения
lr_model = LogisticRegression(solver='liblinear', multi_class='ovr').fit(X_train, y_train)
lda_model = LinearDiscriminantAnalysis().fit(X_train, y_train)
knn_model = KNeighborsClassifier().fit(X_train, y_train)
cart_model = DecisionTreeClassifier().fit(X_train, y_train)
nb_model = GaussianNB().fit(X_train, y_train)
svm_model = SVC().fit(X_train, y_train)

def generate_prediction_plot(model, bounce_rate, exit_rate):
    # Создаем прогноз
    prediction = model.predict([[bounce_rate, exit_rate]])

    # Строим график прогнозирования
    plt.figure(figsize=(8, 6))
    plt.scatter(bounce_rate, exit_rate, c='blue', label='Input Data')
    plt.title('Purchase Prediction')
    plt.xlabel('Bounce Rate')
    plt.ylabel('Exit Rate')
    plt.grid(True)

    # Подписываем прогноз
    if prediction[0] == 1:
        plt.text(bounce_rate, exit_rate, 'Purchase', color='red', fontsize=12, ha='center', va='bottom')
    else:
        plt.text(bounce_rate, exit_rate, 'No Purchase', color='red', fontsize=12, ha='center', va='bottom')

    # Конвертируем график в формат PNG
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()

    # Кодируем изображение в base64 для вставки в HTML
    encoded_img = base64.b64encode(buf.read()).decode('utf-8')
    return f'data:image/png;base64,{encoded_img}'

@app.route('/')
def indexglavn():
    return render_template('indexglavn.html')

@app.route('/indexlearning')
def indexlearning():
    try:
        # Загрузка данных
        data = pd.read_csv('online_shoppers_intention.csv')
        # Анализ датасета
        dataset_shape = data.shape
        dataset_head = data.head()
        dataset_description = data.describe()
        class_distribution = data['Revenue'].value_counts()
        revenue_distribution = data['Revenue'].value_counts()

        # Оценка моделей машинного обучения
        lr_train_accuracy = lr_model.score(X_train, y_train)
        lr_test_accuracy = lr_model.score(X_test, y_test)
        lda_train_accuracy = lda_model.score(X_train, y_train)
        lda_test_accuracy = lda_model.score(X_test, y_test)
        knn_train_accuracy = knn_model.score(X_train, y_train)
        knn_test_accuracy = knn_model.score(X_test, y_test)
        cart_train_accuracy = cart_model.score(X_train, y_train)
        cart_test_accuracy = cart_model.score(X_test, y_test)
        nb_train_accuracy = nb_model.score(X_train, y_train)
        nb_test_accuracy = nb_model.score(X_test, y_test)
        svm_train_accuracy = svm_model.score(X_train, y_train)
        svm_test_accuracy = svm_model.score(X_test, y_test)

        # Возвращаем результаты на веб-страницу
        return render_template('indexlearning.html',
                                dataset_shape=dataset_shape, dataset_head=dataset_head,
                                dataset_description=dataset_description, class_distribution=class_distribution,
                                revenue_distribution=revenue_distribution,
                                lr_train_accuracy=lr_train_accuracy, lr_test_accuracy=lr_test_accuracy,
                                lda_train_accuracy=lda_train_accuracy, lda_test_accuracy=lda_test_accuracy,
                                knn_train_accuracy=knn_train_accuracy, knn_test_accuracy=knn_test_accuracy,
                                cart_train_accuracy=cart_train_accuracy, cart_test_accuracy=cart_test_accuracy,
                                nb_train_accuracy=nb_train_accuracy, nb_test_accuracy=nb_test_accuracy,
                                svm_train_accuracy=svm_train_accuracy, svm_test_accuracy=svm_test_accuracy)
    
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return "An error occurred. Please check the logs."

@app.route('/indexmonitoring')
def indexmonitoring():
    # Загрузка данных
    data = pd.read_csv('online_shoppers_intention.csv')

    # Подсчет количества визитов или продуктовых запросов по месяцам
    monthly_data = data.groupby('Month').size()

    # Создание столбчатой диаграммы с использованием Plotly
    bar_data = go.Bar(x=monthly_data.index, y=monthly_data.values)

    bar_layout = go.Layout(
        title='Столбчатая диаграмма по месяцам',
        xaxis=dict(title='Месяц'),
        yaxis=dict(title='Количество')
    )

    bar_fig = go.Figure(data=[bar_data], layout=bar_layout)

    # Преобразование столбчатой диаграммы в HTML
    bar_plot_div = plot(bar_fig, output_type='div')

    # Подсчет распределения выручки
    revenue_distribution = data['Revenue'].value_counts()

    # Отображение результатов на веб-странице и передача данных
    return render_template('indexmonitoring.html', bar_plot_div=bar_plot_div, revenue_distribution=revenue_distribution)

@app.route('/index')
def index():
    # Загрузка данных
    data = pd.read_csv('online_shoppers_intention.csv')

    # Подсчет количества визитов или продуктовых запросов по месяцам
    monthly_data = data.groupby('Month').size()

    # Создание столбчатой диаграммы с использованием Plotly
    bar_data = go.Bar(x=monthly_data.index, y=monthly_data.values)

    bar_layout = go.Layout(
        title='Столбчатая диаграмма по месяцам',
        xaxis=dict(title='Месяц'),
        yaxis=dict(title='Количество')
    )

    bar_fig = go.Figure(data=[bar_data], layout=bar_layout)

    # Преобразование столбчатой диаграммы в HTML
    bar_plot_div = plot(bar_fig, output_type='div')

    # Подсчет соотношения различных типов посетителей
    visitor_types = data['VisitorType'].value_counts()

    # Создание круговой диаграммы с использованием Plotly
    pie_data = go.Pie(labels=visitor_types.index, values=visitor_types.values)

    pie_layout = go.Layout(
        title='Круговая диаграмма по типу посетителя',
    )

    pie_fig = go.Figure(data=[pie_data], layout=pie_layout)

    # Преобразование круговой диаграммы в HTML
    pie_plot_div = plot(pie_fig, output_type='div')

    # Подсчет количества посетителей или продуктовых запросов по типам трафика
    traffic_data = data.groupby('TrafficType').size()

    # Создание линейной диаграммы с использованием Plotly Express
    line_fig = px.line(traffic_data, x=traffic_data.index, y=traffic_data.values)

    # Преобразование линейной диаграммы в HTML
    line_plot_div = plot(line_fig, output_type='div')

    # Создание данных для точечной диаграммы
    bounce_exit_data = data[['BounceRates', 'ExitRates']]

    # Создание точечной диаграммы с использованием Plotly
    scatter_data = go.Scatter(x=bounce_exit_data['BounceRates'], y=bounce_exit_data['ExitRates'], mode='markers')
    scatter_layout = go.Layout(title='Scatter Plot: Bounce Rates vs. Exit Rates', xaxis=dict(title='Bounce Rates'), yaxis=dict(title='Exit Rates'))
    scatter_fig = go.Figure(data=[scatter_data], layout=scatter_layout)

    # Преобразование точечной диаграммы в HTML
    scatter_plot_div = plot(scatter_fig, output_type='div')

    # Подсчет соотношения выручки и не выручки
    revenue_counts = data['Revenue'].value_counts()

    # Создание круговой диаграммы для выручки
    pie_data_revenue = go.Pie(labels=revenue_counts.index, values=revenue_counts.values)

    pie_layout_revenue = go.Layout(title='Круговая диаграмма по выручке')

    pie_fig_revenue = go.Figure(data=[pie_data_revenue], layout=pie_layout_revenue)

    # Преобразование круговой диаграммы в HTML
    pie_plot_div_revenue = plot(pie_fig_revenue, output_type='div')

    # Отображение результатов на веб-странице и передача данных
    return render_template('index.html', bar_plot_div=bar_plot_div, pie_plot_div=pie_plot_div,
                           line_plot_div=line_plot_div, scatter_plot_div=scatter_plot_div,
                           pie_plot_div_revenue=pie_plot_div_revenue)

@app.route('/indexprediction', methods=['GET', 'POST'])
def indexprediction():
    if request.method == 'POST':
        try:
            bounce_rate = float(request.form['bounce_rate'].replace(',', '.'))
            exit_rate = float(request.form['exit_rate'].replace(',', '.'))
            
            # Создаем прогноз
            prediction_lda = lda_model.predict([[bounce_rate, exit_rate]])
            result_lda = 'Покупатель совершит покупку' if prediction_lda[0] == 1 else 'Покупатель не совершит покупку'
            
            # Генерируем график прогнозирования
            prediction_plot = generate_prediction_plot(lda_model, bounce_rate, exit_rate)

            return render_template('prediction.html', result_lda=result_lda, prediction_plot=prediction_plot)
        except Exception as e:
            print(f"An error occurred: {str(e)}")
            return "An error occurred. Please check the logs."
    else:
        return render_template('indexprediction.html')

if __name__ == '__main__':
    app.run(debug=True)

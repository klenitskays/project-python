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
    # Пути к изображениям
    contact_img_path = 'static/img/contact.png'
    prognoz_img_path = 'static/img/prognoz.jpg'
    monitoring_img_path = 'static/img/monitoring.jpg'
    education_img_path = 'static/img/education.png'

    # Функция для чтения изображения и кодирования его в base64
    def encode_image_to_base64(image_path):
        with open(image_path, "rb") as img_file:
            encoded_img = base64.b64encode(img_file.read()).decode('utf-8')
        return f'data:image/png;base64,{encoded_img}'

    # Кодирование изображений в base64
    contact_img = encode_image_to_base64(contact_img_path)
    prognoz_img = encode_image_to_base64(prognoz_img_path)
    monitoring_img = encode_image_to_base64(monitoring_img_path)
    education_img = encode_image_to_base64(education_img_path)

    # Передача переменных с закодированными изображениями в шаблон
    return render_template('indexglavn.html', monitoring_img=monitoring_img, prognoz_img=prognoz_img, education_img=education_img, contact_img=contact_img)


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
    



# TESTS

# модульные тесты
# import unittest
# from your_module import generate_prediction_plot
# import matplotlib.pyplot as plt

# class TestGeneratePredictionPlot(unittest.TestCase):
#     def test_generate_prediction_plot(self):
#         # Создаем модель для тестирования
#         class TestModel:
#             def predict(self, data):
#                 # Здесь мы имитируем прогноз модели
#                 # В данном случае, всегда возвращаем 1
#                 return [1]

#         model = TestModel()
#         bounce_rate = 0.5
#         exit_rate = 0.4

#         # Вызываем функцию для тестирования
#         plot_html = generate_prediction_plot(model, bounce_rate, exit_rate)

#         # Проверяем, что возвращенное значение не является пустой строкой
#         self.assertNotEqual(plot_html, '')

#         # Проверяем, что график был создан
#         self.assertTrue(isinstance(plt.gcf(), plt.Figure))

# тестирования функций обучения моделей

# class TestModelTraining(unittest.TestCase):
#     def test_data_loading(self):
#         # Check if data is loaded correctly
#         self.assertIsNotNone(X_train)
#         self.assertIsNotNone(X_test)
#         self.assertIsNotNone(y_train)
#         self.assertIsNotNone(y_test)

#     def test_data_splitting(self):
#         # Check if data is split correctly into training and testing sets
#         self.assertEqual(X_train.shape[0] + X_test.shape[0], len(X_train) + len(X_test))
#         self.assertEqual(y_train.shape[0] + y_test.shape[0], len(y_train) + len(y_test))

#     def test_model_training(self):
#         # Check if all models are trained successfully
#         self.assertIsNotNone(lr_model)
#         self.assertIsNotNone(lda_model)
#         self.assertIsNotNone(knn_model)
#         self.assertIsNotNone(cart_model)
#         self.assertIsNotNone(nb_model)
#         self.assertIsNotNone(svm_model)

#     def test_model_accuracy(self):
#         # Check if model accuracies are within expected range
#         lr_train_accuracy = lr_model.score(X_train, y_train)
#         lr_test_accuracy = lr_model.score(X_test, y_test)
#         lda_train_accuracy = lda_model.score(X_train, y_train)
#         lda_test_accuracy = lda_model.score(X_test, y_test)
#         knn_train_accuracy = knn_model.score(X_train, y_train)
#         knn_test_accuracy = knn_model.score(X_test, y_test)
#         cart_train_accuracy = cart_model.score(X_train, y_train)
#         cart_test_accuracy = cart_model.score(X_test, y_test)
#         nb_train_accuracy = nb_model.score(X_train, y_train)
#         nb_test_accuracy = nb_model.score(X_test, y_test)
#         svm_train_accuracy = svm_model.score(X_train, y_train)
#         svm_test_accuracy = svm_model.score(X_test, y_test)

#         self.assertTrue(0 <= lr_train_accuracy <= 1)
#         self.assertTrue(0 <= lr_test_accuracy <= 1)
#         self.assertTrue(0 <= lda_train_accuracy <= 1)
#         self.assertTrue(0 <= lda_test_accuracy <= 1)
#         self.assertTrue(0 <= knn_train_accuracy <= 1)
#         self.assertTrue(0 <= knn_test_accuracy <= 1)
#         self.assertTrue(0 <= cart_train_accuracy <= 1)
#         self.assertTrue(0 <= cart_test_accuracy <= 1)
#         self.assertTrue(0 <= nb_train_accuracy <= 1)
#         self.assertTrue(0 <= nb_test_accuracy <= 1)
#         self.assertTrue(0 <= svm_train_accuracy <= 1)
#         self.assertTrue(0 <= svm_test_accuracy <= 1)

# Для тестирования взаимодействия между маршрутами

# class TestAppRoutes(unittest.TestCase):
#     def setUp(self):
#         app.testing = True
#         self.app = app.test_client()

#     def test_indexglavn_route(self):
#         response = self.app.get('/')
#         self.assertEqual(response.status_code, 200)
#         # Дополнительные проверки на передачу данных и корректный HTML могут быть добавлены здесь

#     def test_indexlearning_route(self):
#         response = self.app.get('/indexlearning')
#         self.assertEqual(response.status_code, 200)
#         # Дополнительные проверки на передачу данных и корректный HTML могут быть добавлены здесь

#     def test_indexmonitoring_route(self):
#         response = self.app.get('/indexmonitoring')
#         self.assertEqual(response.status_code, 200)
#         # Дополнительные проверки на передачу данных и корректный HTML могут быть добавлены здесь

#     def test_index_route(self):
#         response = self.app.get('/index')
#         self.assertEqual(response.status_code, 200)
#         # Дополнительные проверки на передачу данных и корректный HTML могут быть добавлены здесь

#     def test_indexprediction_route(self):
#         response = self.app.get('/indexprediction')
#         self.assertEqual(response.status_code, 200)

# функциональное тестирование

# class TestAppFunctionality(unittest.TestCase):
#     def setUp(self):
#         # Инициализация веб-драйвера Selenium
#         self.driver = webdriver.Chrome()

#     def tearDown(self):
#         # Закрытие веб-драйвера после каждого теста
#         self.driver.quit()

#     def test_indexglavn_page(self):
#         # Тестирование страницы indexglavn
#         self.driver.get('http://127.0.0.1:5000/')
#         # Проверка наличия необходимых элементов на странице, например, изображений
#         contact_img = self.driver.find_element_by_xpath('//*[@id="contact_img"]')
#         prognoz_img = self.driver.find_element_by_xpath('//*[@id="prognoz_img"]')
#         monitoring_img = self.driver.find_element_by_xpath('//*[@id="monitoring_img"]')
#         education_img = self.driver.find_element_by_xpath('//*[@id="education_img"]')
#         self.assertIsNotNone(contact_img)
#         self.assertIsNotNone(prognoz_img)
#         self.assertIsNotNone(monitoring_img)
#         self.assertIsNotNone(education_img)

# if __name__ == '__main__':
#     unittest.main()


if __name__ == '__main__':
    app.run(debug=True)
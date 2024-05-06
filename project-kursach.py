# from flask import Flask, render_template, url_for, redirect
# import pandas as pd
# import numpy as np
# import plotly.graph_objects as go
# from plotly.offline import plot
# import plotly.express as px
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score
# from sklearn.preprocessing import OneHotEncoder

# app = Flask(__name__)

# @app.route('/')
# def indexglavn():
#     return render_template('indexglavn.html')

# from flask import Flask, render_template, url_for, redirect
# import pandas as pd
# import numpy as np
# import plotly.graph_objects as go
# from plotly.offline import plot
# import plotly.express as px
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression  # Изменили импорт
# from sklearn.metrics import accuracy_score
# from sklearn.preprocessing import OneHotEncoder

# app = Flask(__name__)

# @app.route('/')
# def indexglavn():
#     return render_template('indexglavn.html')

# @app.route('/indexlearning')
# def indexlearning():
#     try:
#         # Загрузка данных
#         data = pd.read_csv('online_shoppers_intention.csv')

#         # Вывод информации о данных
#         print("Data information:")
#         print(data.info())
#         print("\nData description:")
#         print(data.describe())

#         # Интерполяция пропущенных значений только для признаков
#         data.interpolate(inplace=True)

#         # Замена пропущенных значений в целевой переменной y
#         data['Revenue'].fillna(1, inplace=True)
#         # Преобразование текстовых значений в числа для целевой переменной 'Revenue'
#         data['Revenue'] = data['Revenue'].map({'TRUE': 1, 'FALSE': 0})

#         # Кодирование категориальных признаков 'Month' и 'VisitorType'
#         encoder = OneHotEncoder(drop='first', sparse_output=False)

#         encoded_features = encoder.fit_transform(data[['Month', 'VisitorType']])
#         encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(['Month', 'VisitorType']))
#         data = pd.concat([data.drop(['Month', 'VisitorType'], axis=1), encoded_df], axis=1)

#         # Создание матрицы признаков X и вектора целей y
#         X = data.drop('Revenue', axis=1)
#         y = data['Revenue']

#         # Разделение данных на обучающий и тестовый наборы
#         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
#         # Обучение модели логистической регрессии
#         model = LogisticRegression(random_state=42)  # Изменили модель
#         model.fit(X_train, y_train)

#         # Прогнозирование на тестовом наборе
#         y_pred = model.predict(X_test)

#         # Оценка производительности модели
#         accuracy = accuracy_score(y_test, y_pred)
#     except Exception as e:
#         # В случае возникновения ошибки выводим сообщение в консоль Flask
#         print(f"An error occurred: {str(e)}")
      
#         # Устанавливаем значение accuracy в None, чтобы передать его в шаблон
#         accuracy = None
    
#         # Анализ датасета
#         dataset_shape = data.shape
#         dataset_head = data.head()
#         dataset_description = data.describe()
#         class_distribution = data['Revenue'].value_counts()
#         revenue_distribution = data['Revenue'].value_counts()

#         # Одномерные графики
#         # Пример одномерной визуализации: гистограмма по атрибуту 'BounceRates'
#         hist_data = go.Histogram(x=data['BounceRates'], name='Bounce Rates')
#         hist_layout = go.Layout(title='Гистограмма Bounce Rates')
#         hist_fig = go.Figure(data=[hist_data], layout=hist_layout)
#         hist_plot_div = plot(hist_fig, output_type='div')

#         # Многомерные графики
#         # Пример многомерной визуализации: scatter plot по атрибутам 'BounceRates' и 'ExitRates'
#         scatter_data = go.Scatter(x=data['BounceRates'], y=data['ExitRates'], mode='markers', name='Bounce Rates vs Exit Rates')
#         scatter_layout = go.Layout(title='Scatter Plot: Bounce Rates vs. Exit Rates', xaxis=dict(title='Bounce Rates'), yaxis=dict(title='Exit Rates'))
#         scatter_fig = go.Figure(data=[scatter_data], layout=scatter_layout)
#         scatter_plot_div = plot(scatter_fig, output_type='div')

#         # Отображение результатов на веб-странице и передача данных
#         return render_template('indexlearning.html', accuracy=accuracy,
#                             dataset_shape=dataset_shape, dataset_head=dataset_head,
#                             dataset_description=dataset_description, class_distribution=class_distribution,
#                             revenue_distribution=revenue_distribution, hist_plot_div=hist_plot_div,
#                             scatter_plot_div=scatter_plot_div)


# @app.route('/indexmonitoring')
# def indexmonitoring():
#     # Загрузка данных
#     data = pd.read_csv('online_shoppers_intention.csv')

#     # Подсчет количества визитов или продуктовых запросов по месяцам
#     monthly_data = data.groupby('Month').size()

#     # Создание столбчатой диаграммы с использованием Plotly
#     bar_data = go.Bar(x=monthly_data.index, y=monthly_data.values)

#     bar_layout = go.Layout(
#         title='Столбчатая диаграмма по месяцам',
#         xaxis=dict(title='Месяц'),
#         yaxis=dict(title='Количество')
#     )

#     bar_fig = go.Figure(data=[bar_data], layout=bar_layout)

#     # Преобразование столбчатой диаграммы в HTML
#     bar_plot_div = plot(bar_fig, output_type='div')

#     # Подсчет распределения выручки
#     revenue_distribution = data['Revenue'].value_counts()

#     # Отображение результатов на веб-странице и передача данных
#     return render_template('indexmonitoring.html', bar_plot_div=bar_plot_div, revenue_distribution=revenue_distribution)



# @app.route('/index')
# def index():
#     # Загрузка данных
#     data = pd.read_csv('online_shoppers_intention.csv')

#     # Подсчет количества визитов или продуктовых запросов по месяцам
#     monthly_data = data.groupby('Month').size()

#     # Создание столбчатой диаграммы с использованием Plotly
#     bar_data = go.Bar(x=monthly_data.index, y=monthly_data.values)

#     bar_layout = go.Layout(
#         title='Столбчатая диаграмма по месяцам',
#         xaxis=dict(title='Месяц'),
#         yaxis=dict(title='Количество')
#     )

#     bar_fig = go.Figure(data=[bar_data], layout=bar_layout)

#     # Преобразование столбчатой диаграммы в HTML
#     bar_plot_div = plot(bar_fig, output_type='div')

#     # Подсчет соотношения различных типов посетителей
#     visitor_types = data['VisitorType'].value_counts()

#     # Создание круговой диаграммы с использованием Plotly
#     pie_data = go.Pie(labels=visitor_types.index, values=visitor_types.values)

#     pie_layout = go.Layout(
#         title='Круговая диаграмма по типу посетителя',
#     )

#     pie_fig = go.Figure(data=[pie_data], layout=pie_layout)

#     # Преобразование круговой диаграммы в HTML
#     pie_plot_div = plot(pie_fig, output_type='div')

#     # Подсчет количества посетителей или продуктовых запросов по типам трафика
#     traffic_data = data.groupby('TrafficType').size()

#     # Создание линейной диаграммы с использованием Plotly Express
#     line_fig = px.line(traffic_data, x=traffic_data.index, y=traffic_data.values)

#     # Преобразование линейной диаграммы в HTML
#     line_plot_div = plot(line_fig, output_type='div')

#     # Создание данных для точечной диаграммы
#     bounce_exit_data = data[['BounceRates', 'ExitRates']]

#     # Создание точечной диаграммы с использованием Plotly
#     scatter_data = go.Scatter(x=bounce_exit_data['BounceRates'], y=bounce_exit_data['ExitRates'], mode='markers')
#     scatter_layout = go.Layout(title='Scatter Plot: Bounce Rates vs. Exit Rates', xaxis=dict(title='Bounce Rates'), yaxis=dict(title='Exit Rates'))
#     scatter_fig = go.Figure(data=[scatter_data], layout=scatter_layout)

#     # Преобразование точечной диаграммы в HTML
#     scatter_plot_div = plot(scatter_fig, output_type='div')

#     # Подсчет соотношения выручки и не выручки
#     revenue_counts = data['Revenue'].value_counts()

#     # Создание круговой диаграммы для выручки
#     pie_data_revenue = go.Pie(labels=revenue_counts.index, values=revenue_counts.values)

#     pie_layout_revenue = go.Layout(title='Круговая диаграмма по выручке')

#     pie_fig_revenue = go.Figure(data=[pie_data_revenue], layout=pie_layout_revenue)

#     # Преобразование круговой диаграммы в HTML
#     pie_plot_div_revenue = plot(pie_fig_revenue, output_type='div')

#     # Отображение результатов на веб-странице и передача данных
#     return render_template('index.html', bar_plot_div=bar_plot_div, pie_plot_div=pie_plot_div,
#                            line_plot_div=line_plot_div, scatter_plot_div=scatter_plot_div,
#                            pie_plot_div_revenue=pie_plot_div_revenue)

# if __name__ == '__main__':
#     app.run(debug=True)










########################################################################################

from flask import Flask, render_template, url_for, redirect
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.offline import plot
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression  # Изменили импорт
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder

app = Flask(__name__)

@app.route('/')
def indexglavn():
    return render_template('indexglavn.html')

'''
from flask import Flask, render_template, url_for, redirect
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.offline import plot
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder

app = Flask(__name__)

@app.route('/')
def indexglavn():
    return render_template('indexglavn.html')
'''

@app.route('/indexlearning')
def indexlearning():
    try:
        # Загрузка данных
        data = pd.read_csv('online_shoppers_intention.csv')

        # Вывод информации о данных
        print("Data information:")
        print(data.info())
        print("\nData description:")
        print(data.describe())

        # Интерполяция пропущенных значений только для признаков
        data.interpolate(inplace=True)

        # Замена пропущенных значений в целевой переменной y
        data['Revenue'].fillna(1, inplace=True)
        # Преобразование текстовых значений в числа для целевой переменной 'Revenue'
        data['Revenue'] = data['Revenue'].map({'TRUE': 1, 'FALSE': 0})

        # Кодирование категориальных признаков 'Month' и 'VisitorType'
        encoder = OneHotEncoder(drop='first', sparse_output=False)

        encoded_features = encoder.fit_transform(data[['Month', 'VisitorType']])
        encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(['Month', 'VisitorType']))
        data = pd.concat([data.drop(['Month', 'VisitorType'], axis=1), encoded_df], axis=1)

        # Создание матрицы признаков X и вектора целей y
        X = data.drop('Revenue', axis=1)
        y = data['Revenue']

        # Проверка наличия пропущенных значений в X_train и y
        print("Number of missing values in X:")
        print(X.isnull().sum())
        print("\nNumber of missing values in y:")
        print(y.isnull().sum())

        # Разделение данных на обучающий и тестовый наборы
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Обучение модели логистической регрессии
        model = LogisticRegression(random_state=42)  # Изменили модель
        model.fit(X_train, y_train)

        # Прогнозирование на тестовом наборе
        y_pred = model.predict(X_test)

        # Оценка производительности модели
        accuracy = accuracy_score(y_test, y_pred)
    except Exception as e:
        # В случае возникновения ошибки выводим сообщение в консоль Flask
        print(f"An error occurred: {str(e)}")
      
        # Устанавливаем значение accuracy в None, чтобы передать его в шаблон
        accuracy = None
    
        # Анализ датасета
        dataset_shape = data.shape
        dataset_head = data.head()
        dataset_description = data.describe()
        class_distribution = data['Revenue'].value_counts()
        revenue_distribution = data['Revenue'].value_counts()

        # Одномерные графики
        # Пример одномерной визуализации: гистограмма по атрибуту 'BounceRates'
        hist_data = go.Histogram(x=data['BounceRates'], name='Bounce Rates')
        hist_layout = go.Layout(title='Гистограмма Bounce Rates')
        hist_fig = go.Figure(data=[hist_data], layout=hist_layout)
        hist_plot_div = plot(hist_fig, output_type='div')

        # Многомерные графики
        # Пример многомерной визуализации: scatter plot по атрибутам 'BounceRates' и 'ExitRates'
        scatter_data = go.Scatter(x=data['BounceRates'], y=data['ExitRates'], mode='markers', name='Bounce Rates vs Exit Rates')
        scatter_layout = go.Layout(title='Scatter Plot: Bounce Rates vs. Exit Rates', xaxis=dict(title='Bounce Rates'), yaxis=dict(title='Exit Rates'))
        scatter_fig = go.Figure(data=[scatter_data], layout=scatter_layout)
        scatter_plot_div = plot(scatter_fig, output_type='div')

        # Отображение результатов на веб-странице и передача данных
        return render_template('indexlearning.html', accuracy=accuracy,
                            dataset_shape=dataset_shape, dataset_head=dataset_head,
                            dataset_description=dataset_description, class_distribution=class_distribution,
                            revenue_distribution=revenue_distribution, hist_plot_div=hist_plot_div,
                            scatter_plot_div=scatter_plot_div)
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

if __name__ == '__main__':
    app.run(debug=True)
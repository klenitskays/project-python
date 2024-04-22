from flask import Flask, render_template
import pandas as pd
import plotly.graph_objects as go
from plotly.offline import plot

app = Flask(__name__)
app.static_folder = 'static'

@app.route('/')
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

    # Отображение результатов на веб-странице и передача данных
    return render_template('index.html', bar_plot_div=bar_plot_div, pie_plot_div=pie_plot_div)

if __name__ == '__main__':
    app.run(debug=True, port=8000, threaded=True)
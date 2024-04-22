from flask import Flask, render_template
import pandas as pd
import plotly.graph_objects as go
from plotly.offline import plot
import plotly.express as px

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
    app.run(debug=True, port=8000, threaded=True)
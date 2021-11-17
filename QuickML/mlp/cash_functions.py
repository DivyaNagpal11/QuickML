import pandas as pd
import numpy as np
import datetime as dt
from collections import Iterable
import itertools
import collections
from math import sqrt
from numpy import array
from scipy import stats
import os
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima_model import ARIMA
import pickle
from calendar import monthrange
import plotly
import plotly.graph_objs as go
import plotly.figure_factory as ff
import time
from sklearn.ensemble.forest import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.stattools import acf,pacf
import pmdarima as pm
from dateutil.relativedelta import relativedelta
from statsmodels.tsa.api import VAR
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.statespace.varmax import VARMAX
from warnings import catch_warnings
from warnings import filterwarnings
from sklearn.metrics import mean_squared_error
from operator import itemgetter

def daily_transformation(subset_df):
    subset_daily = subset_df[['total_payments', 'posted_date', 'Weekday']]
    daily_sum = subset_daily.groupby(['posted_date', 'Weekday']).sum().reset_index()
    daily_sum.posted_date = pd.to_datetime(daily_sum['posted_date'])
    daily_sum = daily_sum.sort_values('posted_date')
    return daily_sum


def weekly_transformation(subset_df):
    subset_daily = subset_df[['total_payments', 'Year', 'WeekNum']]
    weekly_sum = subset_daily.groupby(['Year', 'WeekNum']).sum().reset_index()
    weekly_sum = weekly_sum.sort_values(['Year', 'WeekNum'])
    weekly_sum['week-Year'] = weekly_sum['WeekNum'].astype('str') + '-' + weekly_sum['Year'].astype('str')
    return weekly_sum


def monthly_transformation(subset_df):
    monthly_sum = subset_df[['total_payments', 'MonthYear', 'Year', 'Month']].groupby(
        ['MonthYear', 'Year', 'Month']).sum().reset_index()
    monthly_sum.MonthYear = pd.to_datetime(monthly_sum['MonthYear'])
    monthly_sum = monthly_sum.sort_values('MonthYear')
    monthly_sum['Month_name'] = monthly_sum.MonthYear.dt.month_name()
    monthly_sum.MonthYear = monthly_sum['MonthYear'].dt.strftime('%m-%Y')
    return monthly_sum


def count_of_months(subset_df):
    ah_monthly = subset_df.groupby(['MonthYear', 'billing_entity']).agg({'total_payments': {'tot_payments': 'sum','num_transcation': 'count'}}).reset_index()
    ah_monthly.columns = ['MonthYear', 'billing_entity', 'total_payments', 'num_transaction']
    full_data = ah_monthly.groupby(['billing_entity'])['MonthYear'].count().reset_index()
    full_data.columns = ['billing_entity', 'Count of Months']
    return full_data['Count of Months'][0]


def payments_box_plot(y_column, be_mean_list):
    data = [go.Box(y=y_column, boxpoints='all', name="All BE", jitter=0.3, pointpos=-1.8, selectedpoints=[be_mean_list],
                   marker=dict(size=9, color='purple'))]
    layout = go.Layout(yaxis=dict(title="Avg Of Total Payments", zeroline=False), boxmode='group')
    plot_html = plotly.offline.plot({"data": data, "layout": layout}, output_type='div')
    return plot_html


def timeseries_scatter_plot(monthly_sum1):
    labels = list(monthly_sum1['MonthYear'])
    x = range(0, len(labels))
    x_num = []
    for i in x:
        x_num.append(i)
    rolling = monthly_sum1['total_payments'].rolling(3).mean()
    trace0 = go.Scatter(x=monthly_sum1['MonthYear'], y=rolling, mode='lines',
                        name="Quarterly Rolling Mean", marker=dict(color='#CF152A', line=dict(width=1)))
    trace1 = go.Scatter(x=monthly_sum1['MonthYear'], y=monthly_sum1['total_payments'], mode='lines',
                        name="Monthly Payments", marker=dict(color='#1E8BD9', line=dict(width=1)))
    data = [trace0, trace1]
    layout = go.Layout(go.Layout(yaxis=dict(title="Monthly Payments", zeroline=False),
                                 xaxis=go.layout.XAxis(
                                     title="Month-Year",
                                     tickmode='array',
                                     tickvals=x_num,
                                     ticktext=labels,
                                 ), boxmode='group', width=1050, height=500))
    plot_html = plotly.offline.plot({"data": data, "layout": layout}, output_type='div')
    return plot_html


def timeseries_line_chart_plot(monthly_sum):
    years1 = np.sort(list(set(monthly_sum.Year)))
    subset_start = monthly_sum[monthly_sum.Year == years1[0]]
    import calendar
    months = dict((v, k) for v, k in enumerate(calendar.month_name))
    for k in range(1, 12 - len(subset_start.Month) + 1):
        monthly_sum = monthly_sum.append({'total_payments': 0, 'Month_name': months[k], 'Month': k, 'Year': years1[0]},
                                         ignore_index=True)
    sampling_input = list(months.values())[1::]

    data = []
    for i in range(0, len(years1)):
        df = monthly_sum[monthly_sum.Year == years1[i]].sort_values(['Month']).reset_index(drop=True)
        trace = go.Scatter(x=df.index, y=df['total_payments'], mode='lines',
                           name=str(years1[i]), marker=dict(line=dict(width=1)))
        data.append(trace)

    x_nums = []
    x = range(0, len(sampling_input))
    for i in x:
        x_nums.append(i)
    labels = sampling_input
    layout = go.Layout(go.Layout(yaxis=dict(title="Payments Per Year", zeroline=False),
                                 xaxis=go.layout.XAxis(
                                     title="Month",
                                     tickmode='array',
                                     tickvals=x_nums,
                                     ticktext=labels
                                 ), boxmode='group'))
    plot_html = plotly.offline.plot({"data": data, "layout": layout}, output_type='div')
    return plot_html


def timeseries_bar_plot(monthly_sum):
    monthly_sum = monthly_sum.groupby(['Month_name', 'Month'])['total_payments'].mean().reset_index()
    monthly_sum.sort_values('Month', inplace=True)
    trace0 = go.Bar(x=list(monthly_sum.Month_name), y=monthly_sum['total_payments'],
                    marker=dict(color="purple", line=dict(width=1)))
    data = [trace0]
    layout = go.Layout(go.Layout(yaxis=dict(title="Payments Per Month", zeroline=False),
                                 xaxis=go.layout.XAxis(
                                     title="Month",
                                     tickmode='array',
                                 ), barmode='group'))
    plot_html = plotly.offline.plot({"data": data, "layout": layout}, output_type='div')
    return plot_html


def weekday_transformation(subset_df):
    weekday_sum = subset_df[['total_payments', 'Weekday', 'Weekday_num', 'MonthYear']].groupby(
        ['MonthYear', 'Weekday_num', 'Weekday']).sum().reset_index()
    weekday_sum.MonthYear = pd.to_datetime(weekday_sum['MonthYear'])
    weekday_sum = weekday_sum.sort_values(['MonthYear', 'Weekday_num'])
    weekday_sum.MonthYear = weekday_sum['MonthYear'].dt.strftime('%m-%Y')
    weekday_sum1 = subset_df[['total_payments', 'Weekday', 'Weekday_num', 'MonthYear']].groupby(
        ['MonthYear', 'Weekday_num', 'Weekday']).count().reset_index()
    weekday_sum1.MonthYear = pd.to_datetime(weekday_sum1['MonthYear'])
    weekday_sum1 = weekday_sum1.sort_values(['MonthYear', 'Weekday_num'])
    weekday_sum1.MonthYear = weekday_sum1['MonthYear'].dt.strftime('%m-%Y')
    weekday_sum_merged = pd.merge(weekday_sum, weekday_sum1, on=['MonthYear', 'Weekday_num', 'Weekday'])
    weekday_sum_merged['avg'] = weekday_sum_merged['total_payments_x'] / weekday_sum_merged['total_payments_y']
    monthyears = list(set(weekday_sum.MonthYear))
    weekday_sum_mean = weekday_sum[['total_payments', 'Weekday', 'Weekday_num']].groupby(
        ['Weekday_num', 'Weekday']).mean()
    return weekday_sum_merged, monthyears, weekday_sum, weekday_sum_mean


def wd_bar_plot(weekday_sum_mean):
    x = list(range(0, 8))
    labels = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    trace0 = go.Bar(x=list(range(0, len(weekday_sum_mean.index))), y=weekday_sum_mean['total_payments'],
                    marker=dict(color='purple',line=dict(width=1)))
    data = [trace0]
    layout = go.Layout(go.Layout(yaxis=dict(title="Payments", zeroline=False),
                                 xaxis=go.layout.XAxis(
                                     title="Weekday",
                                     tickmode='array',
                                     tickvals=x,
                                     ticktext=labels
                                 ), barmode='group'))
    plot_html = plotly.offline.plot({"data": data, "layout": layout}, output_type='div')
    return plot_html


def wd_line_chart(weekday_sum, monthyears):
    x = list(range(0, 8))
    labels = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    data = []
    colors = ['blue', 'blueviolet', 'brown', 'burlywood', 'cadetblue',
              'chartreuse', 'chocolate', 'coral', 'cornflowerblue', 'darkblue', 'darkcyan',
              'darkgoldenrod', 'darkgreen', 'darkmagenta', 'darkorange',
              'darkorchid', 'aqua', 'bisque', 'darkred', 'darksalmon', 'darkseagreen',
              'darkslateblue', 'darkturquoise', 'darkviolet', 'deeppink', 'deepskyblue',
              'dimgray', 'dodgerblue', 'firebrick',
              'floralwhite', 'forestgreen', 'fuchsia', 'gainsboro',
              'ghostwhite', 'gold', 'goldenrod', 'gray', 'green',
              'greenyellow', 'hotpink', 'indianred', 'indigo', 'khaki', 'lavenderblush', 'lightcoral', 'lightcyan',
              'lightgray', 'lightpink', 'lightsalmon', 'lightseagreen',
              'lightskyblue', 'lightslategray',
              'lightsteelblue', 'magenta', 'maroon', 'mediumaquamarine',
              'mediumblue', 'mediumorchid', 'mediumpurple',
              'mediumseagreen', 'mediumslateblue', 'mediumspringgreen',
              'mediumturquoise', 'mediumvioletred', 'midnightblue',
              'mintcream', 'mistyrose', 'moccasin', 'navajowhite', 'navy',
              'oldlace', 'olive', 'olivedrab', 'orange', 'orangered',
              'orchid', 'palegoldenrod', 'palegreen', 'paleturquoise',
              'palevioletred', 'papayawhip', 'peachpuff', 'peru', 'pink',
              'plum', 'powderblue', 'purple', 'red', 'rosybrown',
              'royalblue', 'saddlebrown', 'salmon', 'sandybrown',
              'seagreen', 'seashell', 'sienna', 'silver', 'skyblue',
              'slateblue', 'slategray', 'snow', 'springgreen',
              'steelblue']
    for i in range(0, len(monthyears)):
        df = weekday_sum[weekday_sum.MonthYear == monthyears[i]].sort_values(['Weekday_num']).reset_index(drop=True)
        if i < 10:
            trace = go.Scatter(x=df.index, y=df['total_payments'], mode='lines',
                               name=str(monthyears[i]), marker=dict(line=dict(width=1)))
        else:
            trace = go.Scatter(x=df.index, y=df['total_payments'], mode='lines',
                               name=str(monthyears[i]), marker=dict(color=colors[i], line=dict(width=1)))
        data.append(trace)
    layout = go.Layout(go.Layout(yaxis=dict(title="Payments Sum Per Weekday", zeroline=False),
                                 xaxis=go.layout.XAxis(
                                     title="Weekday",
                                     tickmode='array',
                                     tickvals=x,
                                     ticktext=labels
                                 ), boxmode='group'))
    plot_html = plotly.offline.plot({"data": data, "layout": layout}, output_type='div')
    return plot_html


def encounter_transformation(subset_df):
    subset_df[['total_payments', 'encounter_class', 'MonthYear']].groupby(['MonthYear', 'encounter_class']).count()
    encounter_class_sum = subset_df[['total_payments', 'encounter_class', 'MonthYear']].groupby(
        ['MonthYear', 'encounter_class']).sum()
    encounter_class_sum.reset_index(level=0, inplace=True)
    encounter_class_sum.reset_index(level=0, inplace=True)
    encounter_class_pivot = encounter_class_sum.pivot(index='MonthYear', columns='encounter_class',
                                                      values='total_payments')

    encounter_class_pivot.reset_index(level=0, inplace=True)
    encounter_class_pivot['MonthYear'] = pd.to_datetime(encounter_class_pivot['MonthYear'])
    encounter_class_pivot['Month_num'] = encounter_class_pivot['MonthYear'].dt.month
    encounter_class_pivot['Month_name'] = encounter_class_pivot['MonthYear'].dt.month_name()
    encounter_class_pivot = encounter_class_pivot.sort_values(['MonthYear']).reset_index(drop=True)
    encounter_class_pivot.MonthYear = encounter_class_pivot['MonthYear'].dt.strftime('%m-%Y')

    enc_overall = subset_df.groupby(['encounter_class'])['total_payments'].sum().reset_index()
    return encounter_class_pivot, enc_overall


def ec_line_plot(encounter_class_pivot):
    encounter_class_pivot.drop('Month_num', axis=1, inplace=True)
    encounter_class_pivot.fillna(0, inplace=True)
    line_df_columns = list(encounter_class_pivot.columns)
    data = []
    for i in range(0, len(line_df_columns)-2):
        trace = go.Scatter(x=encounter_class_pivot[line_df_columns[0]], y=encounter_class_pivot[line_df_columns[i+1]], mode='lines',
                           name=str(line_df_columns[i+1]), marker=dict(line=dict(width=1)))
        data.append(trace)
    layout = go.Layout(go.Layout(yaxis=dict(title="Total Payments", zeroline=False),
                                 xaxis=go.layout.XAxis(title="Month-Year"), boxmode='group'))
    plot_html = plotly.offline.plot({"data": data, "layout": layout}, output_type='div')
    return plot_html


def ec_bar_plot(enc_overall):
    enc_overall.sort_values(by='total_payments', inplace=True)

    trace0 = go.Bar(x=enc_overall['encounter_class'], y=enc_overall['total_payments'],
                    marker=dict(color='purple',line=dict(width=1)))
    data = [trace0]
    layout = go.Layout(go.Layout(yaxis=dict(title="Total Payments", zeroline=False),
                                 xaxis=go.layout.XAxis(
                                     title="Encounter Class",
                                 ), barmode='group'))
    plot_html = plotly.offline.plot({"data": data, "layout": layout}, output_type='div')
    return plot_html


def ec_footfall_plot(subset_df):
    bar_ec = subset_df.groupby('encounter_class')['footfall'].sum().reset_index()
    bar_ec.sort_values(by='footfall', inplace=True)
    trace0 = go.Bar(x=bar_ec['encounter_class'], y=bar_ec['footfall'],
                    marker=dict(color='purple',line=dict(width=1)))
    data = [trace0]
    layout = go.Layout(go.Layout(yaxis=dict(title="Footfall", zeroline=False),
                                 xaxis=go.layout.XAxis(
                                     title="",
                                 ), barmode='group'))
    plot_html = plotly.offline.plot({"data": data, "layout": layout}, output_type='div')
    return plot_html


def ec_charges_plot(subset_df):
    bar_ec = subset_df.groupby('encounter_class')['total_charge'].sum().reset_index()
    bar_ec.sort_values(by='total_charge', inplace=True)
    trace0 = go.Bar(x=bar_ec['encounter_class'], y=bar_ec['total_charge'],
                    marker=dict(color='purple', line=dict(width=1)))
    data = [trace0]
    layout = go.Layout(go.Layout(yaxis=dict(title="Total Charges", zeroline=False),
                                 xaxis=go.layout.XAxis(
                                     title="",
                                 ), barmode='group'))
    plot_html = plotly.offline.plot({"data": data, "layout": layout}, output_type='div')
    return plot_html


def financial_transformation(subset_df):
    f_data_monthly = subset_df.groupby(['financial_class'])['total_payments'].agg(
        {'total_payments': {'total_payments': 'sum', 'num_payments': 'count'}}).reset_index()
    f_data_monthly.columns = ['financial_class', 'total_payments', 'num_payments']

    f_data_line = subset_df.groupby(['MonthYear', 'financial_class'])['total_payments'].sum().reset_index()
    f_data_line_pivot = f_data_line.pivot(index='MonthYear', columns='financial_class',
                                          values='total_payments').reset_index()
    f_data_line_pivot['MonthYear'] = pd.to_datetime(f_data_line_pivot['MonthYear'])
    f_data_line_pivot.sort_values(by='MonthYear', inplace=True)
    f_data_line_pivot['MonthYear'] = f_data_line_pivot['MonthYear'].dt.strftime('%m-%Y')
    # f_data_line_pivot.set_index('MonthYear', inplace=True)
    f_data_line_pivot.fillna(0, inplace=True)
    return f_data_monthly, f_data_line_pivot


def fc_line_plot(f_data_line_pivot):
    f_data_line_pivot_cols = list(f_data_line_pivot.columns)
    data = []
    colors = ['blue', 'blueviolet', 'brown', 'burlywood', 'cadetblue',
              'chartreuse', 'chocolate', 'coral', 'cornflowerblue', 'darkblue', 'darkcyan',
              'darkgoldenrod', 'darkgreen', 'darkmagenta', 'darkorange',
              'darkorchid', 'aqua', 'bisque', 'darkred', 'darksalmon', 'darkseagreen', 'darkviolet', 'deeppink', 'deepskyblue',
              'dimgray', 'dodgerblue', 'firebrick',
              'floralwhite', 'forestgreen', 'fuchsia',
              'ghostwhite', 'gold', 'goldenrod', 'gray', 'green',
              'greenyellow', 'hotpink', 'indianred', 'indigo', 'khaki', 'lavenderblush', 'lightcoral', 'lightcyan',
              'lightgray', 'lightpink', 'lightsalmon', 'lightseagreen',
              'lightskyblue', 'lightslategray',
              'lightsteelblue', 'magenta', 'maroon',
              'mediumblue', 'mediumorchid', 'mediumpurple',
              'mediumseagreen', 'mediumslateblue', 'mediumspringgreen',
              'mediumturquoise', 'mediumvioletred', 'midnightblue',
              'mintcream', 'mistyrose', 'moccasin', 'navajowhite', 'navy',
              'oldlace', 'olive', 'olivedrab', 'orange', 'orangered',
              'orchid', 'palegoldenrod', 'palegreen', 'paleturquoise',
              'palevioletred', 'papayawhip', 'peachpuff', 'peru', 'pink',
              'plum', 'powderblue', 'purple', 'red', 'rosybrown',
              'royalblue', 'saddlebrown', 'salmon', 'sandybrown',
              'seagreen', 'seashell', 'sienna', 'silver', 'skyblue',
              'slateblue', 'slategray', 'snow', 'springgreen',
              'steelblue']
    for i in range(0, len(f_data_line_pivot_cols)-1):
        if i < 10:
            trace = go.Scatter(x=f_data_line_pivot[f_data_line_pivot_cols[0]],
                               y=f_data_line_pivot[f_data_line_pivot_cols[i + 1]], mode='lines',
                               name=str(f_data_line_pivot_cols[i + 1]),
                               marker=dict(line=dict(width=1)))
        else:
            trace = go.Scatter(x=f_data_line_pivot[f_data_line_pivot_cols[0]],
                               y=f_data_line_pivot[f_data_line_pivot_cols[i + 1]], mode='lines',
                               name=str(f_data_line_pivot_cols[i + 1]), marker=dict(color=colors[i], line=dict(width=1)))
        data.append(trace)

    layout = go.Layout(go.Layout(yaxis=dict(title="Payments Per Month", zeroline=False),
                                 xaxis=go.layout.XAxis(title="Month-Year"), boxmode='group'))
    plot_html = plotly.offline.plot({"data": data, "layout": layout}, output_type='div')
    return plot_html


def fc_bar_plot(f_data_monthly):
    f_data_monthly = f_data_monthly.sort_values(by='total_payments', ascending=False)
    trace0 = go.Bar(x=f_data_monthly['financial_class'], y=f_data_monthly['total_payments'],
                    marker=dict(color='purple',line=dict(width=1)))
    data = [trace0]
    layout = go.Layout(go.Layout(yaxis=dict(title="Total Payments", zeroline=False),
                                 xaxis=go.layout.XAxis(
                                     title="Financial Class",
                                 ), barmode='group'))
    plot_html = plotly.offline.plot({"data": data, "layout": layout}, output_type='div')
    return plot_html


def fc_footfall(subset_df):
    bar_fc = subset_df.groupby('financial_class')['footfall'].sum().reset_index(level=0)
    bar_fc.sort_values(by='footfall', ascending=False, inplace=True)
    trace0 = go.Bar(x=bar_fc['financial_class'], y=bar_fc['footfall'],
                    marker=dict(color='purple',line=dict(width=1)))
    data = [trace0]
    layout = go.Layout(go.Layout(yaxis=dict(title="Footfall", zeroline=False),
                                 xaxis=go.layout.XAxis(
                                     title="",
                                 ), barmode='group'))
    plot_html = plotly.offline.plot({"data": data, "layout": layout}, output_type='div')
    return plot_html


def fc_charges(subset_df):
    bar_fc = subset_df.groupby('financial_class')['total_charge'].sum().reset_index(level=0)
    bar_fc.sort_values(by='total_charge', ascending=False, inplace=True)
    trace0 = go.Bar(x=bar_fc['financial_class'], y=bar_fc['total_charge'],
                    marker=dict(color='purple', line=dict(width=1)))
    data = [trace0]
    layout = go.Layout(go.Layout(yaxis=dict(title="Total Charge", zeroline=False),
                                 xaxis=go.layout.XAxis(
                                     title="",
                                 ), barmode='group'))
    plot_html = plotly.offline.plot({"data": data, "layout": layout}, output_type='div')
    return plot_html


def monthly_transformation_ext_reg(subset_df, xreg):
    monthly_sum = subset_df[[xreg, 'MonthYear', 'Year', 'Month']].groupby(
        ['MonthYear', 'Year', 'Month']).sum().reset_index()
    monthly_sum.MonthYear = pd.to_datetime(monthly_sum['MonthYear'])
    monthly_sum = monthly_sum.sort_values('MonthYear')
    monthly_sum['Month_name'] = monthly_sum.MonthYear.dt.month_name()
    monthly_sum.MonthYear = monthly_sum['MonthYear'].dt.strftime('%m-%Y')
    return monthly_sum


def timeseries_ext_reg(subset_df, monthly_res_xregs, xreg):
    labels = list(monthly_res_xregs['MonthYear'])
    x = range(0, len(labels))
    x_num = []
    for i in x:
        x_num.append(i)
    trace1 = go.Scatter(x=monthly_res_xregs['MonthYear'], y=monthly_res_xregs[xreg], mode='lines',
                        name="Monthly"+xreg, marker=dict(color='#1E8BD9', line=dict(width=1)))
    data = [trace1]
    layout = go.Layout(go.Layout(yaxis=dict(title="Monthly Aggregated", zeroline=False),
                                 xaxis=go.layout.XAxis(
                                     title="Month-Year",
                                     tickmode='array',
                                     tickvals=x_num,
                                     ticktext=labels,
                                 ), boxmode='group', width=1050, height=500))
    plot_html = plotly.offline.plot({"data": data, "layout": layout}, output_type='div')
    return plot_html


def ff_lagged_plot(ff_subset_df, monthly_data, xreg):
    ff_monthly_df = ff_subset_df.groupby(['MonthYear'])[xreg].sum().reset_index(level=0)
    ff_monthly_df['MonthYear'] = pd.to_datetime(ff_monthly_df['MonthYear'])
    ff_monthly_df.sort_values('MonthYear', inplace=True)

    lags = 3
    for i in range(1, lags + 1):
        a = 'lag_' + str(i)
        ff_monthly_df[a] = ff_monthly_df[xreg].shift(i)

    ff_monthly_df.bfill(inplace=True)

    ff_monthly_df['MonthYear'] = pd.to_datetime(ff_monthly_df['MonthYear'])

    tot = monthly_data[['MonthYear', 'total_payments']]
    tot['MonthYear'] = pd.to_datetime(tot['MonthYear'])
    ff_monthly_df = pd.merge(ff_monthly_df, tot, on='MonthYear', how='inner')

    ff_monthly_df.set_index('MonthYear', inplace=True)
    ff_monthly_df.drop(xreg, axis=1, inplace=True)
    cols = ff_monthly_df.columns

    # scale the features
    mms = MinMaxScaler()
    ff_monthly_scaled = pd.DataFrame(mms.fit_transform(ff_monthly_df))
    ff_monthly_scaled.columns = cols.copy()
    ff_monthly_scaled['MonthYear'] = ff_monthly_df.index.copy()

    trace1 = {
            "name": "Total Payments",
            "type": "scatter",
            "x": ff_monthly_scaled['MonthYear'],
            "y": ff_monthly_scaled["total_payments"],
            "xaxis": "x1",
            "yaxis": "y1",
            "marker": {"color": "salmon"},
            'mode': 'lines+markers',
        }
    trace2 = {
            "name": xreg + "-Lag 1",
            "type": "scatter",
            "x": ff_monthly_scaled['MonthYear'],
            "y": ff_monthly_scaled["lag_1"],
            "xaxis": "x1",
            "yaxis": "y1",
            'line': dict(color='teal', width=2, dash='dot')
        }

    trace3 = {
        "type": "scatter",
        "x": ff_monthly_scaled['MonthYear'],
        "y": ff_monthly_scaled["total_payments"],
        "xaxis": "x1",
        "yaxis": "y2",
        "marker": {"color": "salmon"},
        'showlegend': False,
        'mode': 'lines+markers',
    }
    trace4 = {
        "name": xreg + "-Lag 2",
        "type": "scatter",
        "x": ff_monthly_scaled['MonthYear'],
        "y": ff_monthly_scaled["lag_2"],
        "xaxis": "x1",
        "yaxis": "y2",
        'line': dict(color='royalblue', width=2, dash='dot')
    }

    trace5 = {
        "type": "scatter",
        "x": ff_monthly_scaled['MonthYear'],
        "y": ff_monthly_scaled["total_payments"],
        "xaxis": "x1",
        "yaxis": "y3",
        "marker": {"color": "salmon"},
        'showlegend': False,
        'mode': 'lines+markers',
    }
    trace6 = {
        "name": xreg + "-Lag 3",
        "type": "scatter",
        "x": ff_monthly_scaled['MonthYear'],
        "y": ff_monthly_scaled["lag_3"],
        "xaxis": "x1",
        "yaxis": "y3",
        'line': dict(color='goldenrod', width=2, dash='dot')
    }

    data = [trace1, trace2, trace3, trace4, trace5, trace6]
    layout = {"xaxis1": {
                   "anchor": "y3",
                   "domain": [0.0, 1.0]},
              "yaxis1": {
                  "anchor": "free",
                  "domain": [0.70, 0.9999999999999999],
                  "position": 0.0},
              "yaxis2": {
                  "anchor": "free",
                  "domain": [0.35, 0.65],
                  "position": 0.0},
              "yaxis3": {
                  "anchor": "free",
                  "domain": [0.0, 0.30],
                  "position": 0.0},
              'showlegend': True,
              'height': 600,
              }

    plot_html = plotly.offline.plot({"data": data, "layout": layout}, output_type='div')
    return plot_html


def ff_per_month(ff_subset_df, monthly_data, xreg):
    ff_monthly = ff_subset_df.groupby(['MonthYear'])[xreg].sum().reset_index()
    ff_monthly['MonthYear'] = pd.to_datetime(ff_monthly['MonthYear'])
    ff_monthly.sort_values(by='MonthYear', axis=0, inplace=True)

    ff_monthly['lag1'] = ff_monthly[xreg].shift(1)  # after 30 days
    ff_monthly['lag2'] = ff_monthly[xreg].shift(2)  # after 60 days
    ff_monthly['lag3'] = ff_monthly[xreg].shift(3)  # after 90 days
    # ff_monthly['lag4'] = ff_monthly[xreg].shift(4)  # after 120 days
    # ff_monthly['lag5'] = ff_monthly[xreg].shift(5)  # after 120 days
    ff_monthly.bfill(inplace=True)

    ff_monthly_or = ff_monthly.copy()
    tot = monthly_data[['MonthYear', 'total_payments']]
    tot['MonthYear'] = pd.to_datetime(tot['MonthYear'])

    ff_monthly_or = pd.merge(ff_monthly_or, tot, on='MonthYear', how='inner')
    ff_monthly_or.set_index('MonthYear', inplace=True)
    ff_monthly_or.drop(xreg, axis=1, inplace=True)
    cols = ff_monthly_or.columns
    lags_corr = {}
    for i in range(0, len(cols) - 1):
        c = ff_monthly_or[[cols[i], 'total_payments']].corr()
        lags_corr[cols[i]] = c['total_payments'][0]

    trace1 = go.Scatter(x=list(lags_corr.keys()), y=list(lags_corr.values()), mode='lines+markers',
                        name="Correlation with payments", marker=dict(color='#1E8BD9', line=dict(width=1)))
    data = [trace1]
    layout = go.Layout(go.Layout(yaxis=dict(title="Correlation", zeroline=False),
                                 xaxis=go.layout.XAxis(
                                     title="Lags",
                                 ), boxmode='group',height=300))
    plot_html = plotly.offline.plot({"data": data, "layout": layout}, output_type='div')
    return plot_html


def ff_xreg_correlation(subset_df, monthly_data, xreg):
    data = subset_df.groupby(['MonthYear', 'financial_class'])[xreg].sum().reset_index()
    data['MonthYear'] = pd.to_datetime(data['MonthYear'])
    data.sort_values(by='MonthYear', axis=0, inplace=True)

    fc_pivot = data.pivot(index='MonthYear', columns='financial_class', values=xreg).reset_index()
    fc_pivot.set_index('MonthYear', inplace=True)

    fc_pivot.fillna(0, inplace=True)
    xreg_corr = pd.DataFrame()

    for i in range(1, 4):
        tmp = fc_pivot.shift(i)
        tmp.bfill(inplace=True)
        tmp.reset_index(inplace=True)
        # add payments data
        tot = monthly_data[['MonthYear', 'total_payments']]
        tot['MonthYear'] = pd.to_datetime(tot['MonthYear'])

        fc_merge = pd.merge(tot, tmp, on='MonthYear', how='inner')
        fc_merge.set_index('MonthYear', inplace=True)

        fc_corr = pd.DataFrame(fc_merge.corr()['total_payments'])
        fc_corr.rename(columns={'total_payments': 'Lag' + str(i)}, inplace=True)

        xreg_corr = pd.concat([xreg_corr, fc_corr], axis=1)
    return xreg_corr.transpose()


def acf_plot(corr_df):
    trace0 = go.Bar(x=list(range(0, len(corr_df))), y=corr_df['ACF'], width=0.25,
                    marker=dict(color='purple', line=dict(width=1)))
    trace1 = go.Scatter(x=list(range(0, len(corr_df))), y=corr_df['upper_acf']-corr_df['ACF'], mode='none', fill='tonexty',
                        fillcolor='rgba(128, 128, 128,0.15)', showlegend=False, marker=dict(line=dict(width=1)))
    trace2 = go.Scatter(x=list(range(0, len(corr_df))), y=corr_df['lower_acf']-corr_df['ACF'], mode='none', fill='tonexty',
                        fillcolor='rgba(128, 128, 128,0.15)', showlegend=False, marker=dict(line=dict(width=1)))
    data = [trace0, trace1, trace2]
    layout = go.Layout(go.Layout(yaxis=dict(title="ACF", zeroline=False),
                                 xaxis=go.layout.XAxis(
                                     title="Lags",
                                 ), barmode='group'))
    plot_html = plotly.offline.plot({"data": data, "layout": layout}, output_type='div')
    return plot_html


def pacf_plot(corr_df):
    trace0 = go.Bar(x=list(range(0, len(corr_df))), y=corr_df['PACF'], width=0.25,
                    marker=dict(color='purple', line=dict(width=1)))
    trace1 = go.Scatter(x=list(range(0, len(corr_df))), y=corr_df['upper_pacf']-corr_df['PACF'], mode='none', fill='tonexty',
                        fillcolor='rgba(128, 128, 128,0.10)', showlegend=False, marker=dict(line=dict(width=1)))
    trace2 = go.Scatter(x=list(range(0, len(corr_df))), y=corr_df['lower_pacf']-corr_df['PACF'], mode='none', fill='tonexty',
                        fillcolor='rgba(128, 128, 128,0.15)', showlegend=False, marker=dict(line=dict(width=1)))
    data = [trace0, trace1, trace2]
    layout = go.Layout(go.Layout(yaxis=dict(title="PACF", zeroline=False),
                                 xaxis=go.layout.XAxis(
                                     title="Lags",
                                 ), barmode='group'))
    plot_html = plotly.offline.plot({"data": data, "layout": layout}, output_type='div')
    return plot_html


########################### Best Lag Code ###################################

# generates lagged features

def monthly_xreg(subset_df,lags):
    subset_df_new=subset_df.copy()
    subset_df_new['MonthYear']=pd.to_datetime(subset_df_new['MonthYear'])
    x_data = subset_df_new.groupby(['MonthYear'])['footfall','total_charge'].sum().reset_index()
    x_data.set_index('MonthYear', inplace=True)
    x_data = lagged_ts(x_data, lags) # as NaN needs to be filled with lagged values
    return x_data


def lagged_features(series, subset_df, lags):
    ### Fetch external regressors- weekdays, Payor wise PMI, Payor wise footfall,Payor wise charges
    ff_ch = monthly_xreg(subset_df, lags)

    df = pd.merge(pd.DataFrame(series), ff_ch, on=ff_ch.index, how='inner')
    df.rename(columns={'key_0': 'MonthYear'}, inplace=True)
    df.set_index('MonthYear', inplace=True)
    return df


def infer_lag_corr(df_lag3, df_lag2, var):
    corr_diff = np.abs(df_lag3.corr()[var]) - np.abs(df_lag2.corr()[var])
    # Initially lag setting to 2
    ff_lag = 2
    ch_lag = 2
    if corr_diff['footfall'] > 0:
        ff_lag = 3
    if corr_diff['total_charge'] > 0:
        ch_lag = 3

    # If both ff and ch lags are same
    if ff_lag == ch_lag:
        final_lag = ff_lag
        feature = 'ff_ch'
    # If ff lag corr is greater than ch lag corr
    elif np.abs(corr_diff['footfall']) > np.abs(corr_diff['total_charge']):
        final_lag = ff_lag
        feature = 'ff'
    else:
        final_lag = ch_lag
        feature = 'ch'
    return final_lag, feature


def get_best_lag(subset_df, be_name):
    All_BE_lag_dic = {}
    monthly_data = monthly_transformation(subset_df)
    var = 'total_payments'
    series = monthly_data[var]
    series.index = pd.DatetimeIndex(monthly_data.MonthYear)

    df_lag2 = lagged_features(series, subset_df, 2)
    df_lag3 = lagged_features(series, subset_df, 3)

    final_lag, feature = infer_lag_corr(df_lag3, df_lag2, var)

    # All_BE_lag_dic[be_name] = (final_lag, feature)
    return final_lag


def wday_per_month_new(subset_df, n_periods):
    ## daily dates generation for next 5 years
    new_df = pd.DataFrame(columns=['posted_date', 'weekday'])
    start_date = subset_df['posted_date'].min()
    start_date = dt.date(start_date.year, start_date.month, day=1)
    end_date = subset_df['posted_date'].max() + relativedelta(months=4)
    end_date = dt.date(end_date.year, end_date.month, day=1)
    tmp_dt1 = pd.date_range(start_date, end_date, freq='D').tolist()
    # remove extra date
    tmp_dt1 = tmp_dt1[:len(tmp_dt1) - 1]

    # make dateframe for count of each weekday
    new_df['posted_date'] = new_df['posted_date'].append(pd.Series((v for v in tmp_dt1)))
    new_df['weekday'] = new_df['posted_date'].apply(lambda r: r.weekday())
    new_df['Month'] = new_df['posted_date'].dt.month
    new_df['Year'] = new_df['posted_date'].dt.year
    new_df['MonthYear'] = new_df['Month'].astype('str') + '-' + new_df['Year'].astype('str')
    new_wday = new_df.groupby(['MonthYear', 'weekday'])['Month'].count().reset_index()
    new_wday.columns = ['MonthYear', 'Weekday_num', 'Count']
    new_wday['MonthYear'] = pd.to_datetime(new_wday['MonthYear'])
    new_wday.sort_values(by=['MonthYear', 'Weekday_num'], inplace=True)
    wday = new_wday.pivot(index='MonthYear', columns='Weekday_num', values='Count').reset_index()

    # 1. working days
    wday['Working_days'] = wday[0] + wday[1] + wday[2] + wday[3] + wday[4]
    # 2 weekend days
    wday['Weekends_days'] = wday[5] + wday[6]
    # wday.drop([5,6],axis=1,inplace=True)
    wday.columns = ['MonthYear', 'n_Mon', 'n_Tue', 'n_Wed', 'n_Thu', 'n_Fri', 'n_Sat', 'n_Sun', 'n_Working_days',
                    'n_Weekends_days']
    # sort data
    wday['MonthYear'] = pd.to_datetime(wday['MonthYear'])
    wday.sort_values(by=['MonthYear'], inplace=True)
    length = subset_df['MonthYear'].nunique() + n_periods  # Trim extra data
    wday = wday[0:length]
    wday.set_index('MonthYear', inplace=True)
    # segregate original data and unseen data for weekdays
    xreg_wday = wday.head(-n_periods)
    pred_xreg_wday = wday.tail(n_periods)
    return wday, xreg_wday, pred_xreg_wday


def make_forcasted_dates(subset_df,n_periods):
    tmp=[]
    st_dt=subset_df['posted_date'].max()
    for i in range(0,n_periods):
        st_dt=st_dt+relativedelta(months=1)
        st_dt=dt.date(st_dt.year,st_dt.month,day=1)
        tmp.append(st_dt)
    return tmp


def convert_df_percentage(df):
    ####  Replace negative values with previous month payments/charges
    cols = df.columns
    for i in cols:
        if any(df[i] < 0):
            for j in range(0, len(df[i])):
                if df[i][j] < 0:
                    df[i][j] = np.nan
    df = df.bfill()

    #### convert in percentage  ########
    df['Sum'] = df.apply(lambda r: r.sum(), axis=1)
    tot_ff = df['Sum'].copy().rename('Sum', inplace=True)
    df = df.apply(lambda r: round((r / tot_ff) * 100, 2))
    df.drop('Sum', axis=1, inplace=True)
    return df


def lagged_ts(ts, lags):
    ts = ts.shift(lags)
    ts.bfill(inplace=True)
    # replace NaN values
    return ts


def segregate_act_pred(df, n_periods):
    length = len(df)-n_periods
    actual = df[0:length]
    predicted = df[length:]
    return actual, predicted


def ec_PMI(subset_df, n_periods, lags, percent_flag):
    pmi_data = subset_df.groupby(['MonthYear', 'financial_class'])['total_payments'].sum().reset_index()

    # create into pivot
    pmi_data = pmi_data.pivot(index='MonthYear', columns='financial_class', values='total_payments').reset_index()
    pmi_data['MonthYear'] = pd.to_datetime(pmi_data['MonthYear'])
    pmi_data.sort_values(by='MonthYear', inplace=True)
    pmi_data.set_index('MonthYear', inplace=True)
    pmi_data.fillna(0, inplace=True)
    # convert to percentage
    if percent_flag != 0:
        pmi_data = convert_df_percentage(pmi_data)

    ##### drop financial classes with 0 contribution
    cols = pmi_data.columns
    for i in range(len(cols)):
        if (pmi_data[cols[i]] == 0).all() == True:
            pmi_data.drop(cols[i], axis=1, inplace=True)
        ##### Add forecasted months to pmi data
    tmp = make_forcasted_dates(subset_df, n_periods)
    new_df = pd.DataFrame(columns=['MonthYear'])
    new_df['MonthYear'] = new_df['MonthYear'].append(pd.Series((v for v in tmp)))
    cols = pmi_data.columns

    ########Shift time series
    for i in range(0, len(cols)):
        new_df[cols[i]] = np.nan
    new_df.set_index('MonthYear', inplace=True)
    pmi_data = pmi_data.append(new_df)
    # as NaN needs to be filled with lagged values
    pmi_data = lagged_ts(pmi_data, lags)

    ##### OUTLIER REJECTION
    # pmi_data_or=outlier_rejection_df_z(pmi_data) #z score rejection
    # pmi_data_or=outlier_rejection_df_quantile(pmi_data) #Quantile based rejection
    ##### 4 Hypothesis Data
    # 1. All financial classes PMI

    combined_xreg_pmi = pmi_data.copy()
    act_xreg_pmi, pred_xreg_pmi = segregate_act_pred(combined_xreg_pmi, n_periods)
    return act_xreg_pmi, pred_xreg_pmi


def ec_Xreg(subset_df, x_agg_var, n_periods, lags, percent_flag):
    subset_df_new = subset_df.copy()
    subset_df_new['MonthYear'] = pd.to_datetime(subset_df_new['MonthYear'])
    x_data = subset_df_new.groupby(['MonthYear', 'financial_class'])[x_agg_var].sum().reset_index()

    # rename financial classes and encounter classes based on charges and footfall data
    if x_agg_var == 'footfall':
        x_data['financial_class'] = x_data['financial_class'].apply(lambda r: str(r) + '_ff')
    elif x_agg_var == 'total_charge':
        x_data['financial_class'] = x_data['financial_class'].apply(lambda r: str(r) + '_ch')

    # convert to pivot
    x_data = x_data.pivot(index='MonthYear', columns='financial_class', values=x_agg_var)
    x_data.fillna(0, inplace=True)
    # convert percentage
    if percent_flag != 0:
        x_data = convert_df_percentage(x_data)
        ##### drop financial classes with 0 contribution
    cols = x_data.columns
    for i in range(len(cols)):
        if (x_data[cols[i]] == 0).all() == True:
            x_data.drop(cols[i], axis=1, inplace=True)
        ##### Add forecasted months to pmi data
    tmp = make_forcasted_dates(subset_df_new, n_periods)
    new_df = pd.DataFrame(columns=['MonthYear'])
    new_df['MonthYear'] = new_df['MonthYear'].append(pd.Series((v for v in tmp)))
    cols = x_data.columns
    ##### Shift time series
    for i in range(0, len(cols)):
        new_df[cols[i]] = np.nan
    new_df.set_index('MonthYear', inplace=True)
    x_data = x_data.append(new_df)
    x_data = lagged_ts(x_data, lags)  # as NaN needs to be filled with lagged values

    ##### segregate data
    combined_xreg_x = x_data.copy()
    combined_xreg_x.reset_index(inplace=True)
    combined_xreg_x['MonthYear'] = pd.to_datetime(combined_xreg_x['MonthYear'])
    combined_xreg_x.sort_values('MonthYear', inplace=True)
    combined_xreg_x.set_index('MonthYear', inplace=True)
    act_xreg_x, pred_xreg_x = segregate_act_pred(combined_xreg_x, n_periods)
    return act_xreg_x, pred_xreg_x


def train_test_split(data, n_test):
    return data[:-n_test], data[-n_test:]


def mape(actual, predicted):
    return np.mean(np.abs(np.subtract(actual, predicted) / actual)) * 100


def smape1(actual, predicted):
    return 2 * np.mean(np.abs(np.subtract(actual, predicted) / (actual + predicted))) * 100


def create_test(df, data, dt, n_periods, rem):
    var_mape = {}
    # Segregating features and creating dataframe with payor features
    xregressor = data.filter(like='n_')
    cols = [col for col in df.columns if col not in xregressor.columns]
    d_vars = data[cols].iloc[:, 1::]
    var_flag = 0
    if n_periods == 3 and rem == 3:
        val_start = dt + relativedelta(months=1)
        val_end = dt + relativedelta(months=n_periods)
        val = df[val_start:val_end]
        x_test = val.iloc[:, 1:]
    elif rem < 3:
        val_start = dt + relativedelta(months=1)
        val_end = dt + relativedelta(months=rem)
        val = df[val_start:val_end]
        x_test = val.iloc[:, 1:]
    elif n_periods != 3 and rem == 3:
        var_flag = 1
        val_start = dt + relativedelta(months=1)
        val_end = dt + relativedelta(months=n_periods)
        val = df[val_start:val_end]
        x_test = val.iloc[:, 1:]

        # remaining month regressors need to be predicted using VAR
        mod = VAR(d_vars)
        model_fit = mod.fit()
        pred_xregs = model_fit.forecast(model_fit.y, steps=3)

        val1 = df[val_end + relativedelta(months=1):val_end + relativedelta(months=3 - n_periods)]
        wday_features = val1.filter(like='n_')
        actual_regressor_features = val1[cols].iloc[:, 1::]
        d = pd.DataFrame(pred_xregs)
        d.columns = d_vars.columns

        test_month_xregs = d.iloc[n_periods::]  # Getting forecast features from var for remaining months
        test_month_xregs.index = val1.index.copy()

        comb = pd.concat([test_month_xregs, wday_features], axis=1)

        x_test = x_test.append(comb)

        # check mape
        var_mape = {}
        for n in range(len(val1)):
            smape = {}
            for i, name in enumerate(d_vars.columns):
                smape[name] = smape1(actual_regressor_features.iloc[n][name], d.iloc[n + 1][i])

            var_mape[n] = smape
    return x_test, var_flag, var_mape

############################################ Arima Model ##########################################################


def arima_model_with_xregs_mult(df1, var, regressors):
    model = pm.auto_arima(df1[[var]], exogenous=df1[regressors],
                          start_p=0, start_q=0, test='adf', max_p=5, max_q=5, m=1, start_P=0, seasonal=False, d=None,
                          D=0, trace=True, error_action='ignore', suppress_warnings=True, stepwise=True)
    return model


def arima_model(df1, var):
    model = pm.auto_arima(df1[var], start_p=0, start_q=0,
                          test='adf',  # use adftest to find optimal 'd'
                          max_p=7, max_q=7,  # maximum p and q
                          m=1,  # frequency of series
                          d=None,  # let model determine 'd'
                          seasonal=False,  # No Seasonality
                          start_P=0,
                          start_Q=0,
                          D=0,
                          trace=True,
                          error_action='ignore',
                          suppress_warnings=True,
                          stepwise=True)
    return model


def arima_model_with_xregs(df1, var, regressor):
    model = pm.auto_arima(df1[[var]], exogenous=df1[[regressor]], start_p=0, start_q=0,test='adf',max_p=7, max_q=7, m=1,
                          start_P=0, seasonal=False, d=None, D=0, trace=True, error_action='ignore',
                          suppress_warnings=True, stepwise=True)
    try:
        # never show warnings
        with catch_warnings():
            filterwarnings("ignore")
    except:
        error = None
    return model


def run_best_model_cv(train2, val1, var, reg, n_periods):
    if isinstance(reg, str):
        if reg == 'None':
            model = arima_model(train2, var)
            fitted, confint = model.predict(n_periods=n_periods, return_conf_int=True)
        else:
            model = arima_model_with_xregs(train2, var, reg)
            fitted, confint = model.predict(n_periods=n_periods, exogenous=array(val1[reg]).reshape(-1,1),
                                            return_conf_int=True)
    else:
        model = arima_model_with_xregs_mult(train2, var, reg)
        fitted, confint = model.predict(n_periods=n_periods, exogenous=val1[reg], return_conf_int=True)
    return model, fitted, confint


def run_arima(x_train, y_train, x_test, n_periods_test, var):
    reg = ['n_Mon', 'n_Tue', 'n_Wed', 'n_Thu', 'n_Fri', 'n_Sat', 'n_Sun',
           'n_Working_days', 'n_Weekends_days']
    train2 = pd.concat([y_train, x_train], axis=1)
    model, fitted, confint = run_best_model_cv(train2, x_test, var, reg, n_periods_test)
    return(model,fitted, confint)


def run_arima_null(x_train, y_train, x_test, n_periods_test, var):
    train2 = pd.concat([y_train, x_train], axis=1)
    model, fitted, confint = run_best_model_cv(train2, x_test, var, 'None', n_periods_test)
    return model, fitted, confint


def get_arima_best(x_train, y_train, y_test, x_test, var, rem_data):
    model_list = []
    model_arima_reg, fitted_arima_reg, confint_arima_reg = run_arima(x_train, y_train, x_test, rem_data, var)
    model_arima_null, fitted_arima_null, confint_arima_null = run_arima_null(x_train, y_train, x_test, rem_data, var)
    print("fitted arima reg",fitted_arima_reg)
    print("fitted arima null", fitted_arima_null)
    model_list.append(['model_arima_reg', mape(y_test, fitted_arima_reg)])

    model_list.append(['model_arima_null', mape(y_test, fitted_arima_null)])
    model_list.sort(key=lambda tup: tup[1])
    best_model_name = model_list[0][0]
    if best_model_name == 'model_arima_reg':
        return (best_model_name, model_arima_reg, fitted_arima_reg, confint_arima_reg)
    else:
        return (best_model_name, model_arima_null, fitted_arima_null, confint_arima_null)


############################# Random Forest #############################################
def pred_ints(model, x_test, percent=95):
    err_down = []
    err_up = []
    for x in range(len(x_test)):
        preds = []
        for pred in model.estimators_:
            preds.append(pred.predict(np.array(x_test.iloc[x]).reshape(1, -1))[0])
        quants = np.percentile(preds, [25, 50, 75])
        iqr = quants[2] - quants[0]
        lower_bound = quants[0] - (1.5 * iqr)
        upper_bound = quants[2] + (1.5 * iqr)
        updated_preds = []
        for i in preds:
            if (i > lower_bound) and (i < upper_bound):
                updated_preds.append(i)
        if type(updated_preds) == type(None) or len(updated_preds) == 0:
            err_down.append(np.percentile(preds, (100 - percent) / 2.))
            err_up.append(np.percentile(preds, 100 - (100 - percent) / 2.))
        else:
            err_down.append(np.percentile(updated_preds, (100 - percent) / 2.))
            err_up.append(np.percentile(updated_preds, 100 - (100 - percent) / 2.))

    # make dataframe of confidence interval points
    confint = np.vstack((err_down, err_up)).T
    return (confint)


def rf_feature_importance(rgr, x_train):
    feature_importance = list(rgr.feature_importances_)
    features = list(x_train.columns)
    feature_df = pd.DataFrame(data=[features,feature_importance])
    feature_df = feature_df.transpose()
    feature_df.columns = ['Feature', 'Feature Importance']
    feature_df = feature_df.sort_values(by='Feature Importance', ascending=False)
    trace0 = go.Bar(y=feature_df['Feature'], x=feature_df['Feature Importance'], orientation='h', width=0.8,
                    marker=dict(color='purple', line=dict(width=1)))
    data = [trace0]
    layout = go.Layout(go.Layout(yaxis=dict(zeroline=False, autorange="reversed"),
                                 xaxis=go.layout.XAxis(
                                     title="Feature Importance",
                                 ),margin=go.Margin(l=200, r=30), barmode='group', bargap=0.50, height=600))
    plot_html = plotly.offline.plot({"data": data, "layout": layout}, output_type='div')
    return plot_html


def random_forest(x_train, y_train, x_test):
    # build our RF model
    RF_Model = RandomForestRegressor(n_estimators=100, max_features=1, oob_score=True, random_state=1)

    # Fit the RF model with features and labels.
    rf_model = RF_Model.fit(x_train, y_train)

    # Creating feature importance map
    fimp = {}
    for i, name in enumerate(x_train.columns):
        fimp[name] = rf_model.feature_importances_[i]

    fimp_sorted = collections.OrderedDict(sorted(fimp.items(), key=itemgetter(1), reverse=True))
    fitted = rf_model.predict(x_test)

    percent = 95
    confint = pred_ints(rf_model, x_test, percent)

    return rf_model, fitted, fimp_sorted, confint

############################### Varmax Functions ###########################

def varmax_model(trainV, x_test, rem_data):
    xregressor = trainV.filter(like='n_')
    xreg_cols = xregressor.columns
    cols = [col for col in trainV.columns if col not in xregressor.columns]
    x_train = trainV[cols]
    sc = StandardScaler()
    x_train_f = sc.fit_transform(x_train)
    xreg_train = xregressor[xreg_cols]

    p = 1
    q = 1

    mod = VARMAX(x_train_f, order=(p, q), exog=np.array(xreg_train))
    model = mod.fit(maxiter=1000, disp=False)
    xreg_test = x_test.filter(like='n_')
    xreg_test = xreg_test[xreg_cols]

    fitted = model.forecast(rem_data, exog=xreg_test)
    fitted_f = sc.inverse_transform(fitted)
    fitted = fitted_f[:, 0]
    confint2 = np.array(0)
    for i in range(1, rem_data + 1):
        conf = model.get_prediction((len(trainV) + (i - 1)), exog=xreg_test[:i])
        confint = conf.conf_int(alpha=0.05)
        c1 = sc.inverse_transform([confint[0][0:x_train.shape[1]]])[0][0]
        c2 = sc.inverse_transform([confint[0][x_train.shape[1]::]])[0][0]
        confint1 = np.vstack((c1, c2)).T
        confint2 = np.append(confint2, confint1)
    confint_f = confint2[1::].reshape(rem_data, 2)
    return model, fitted, confint_f


def varmax_model_test(trainV, x_test, rem_data):
    xregressor = trainV.filter(like='n_')
    xreg_cols = xregressor.columns
    cols = [col for col in trainV.columns if col not in xregressor.columns]
    x_train = trainV[cols]
    sc = StandardScaler()
    x_train_f = sc.fit_transform(x_train)
    xreg_train = xregressor[xreg_cols]

    p = 1
    q = 1

    mod = VARMAX(x_train_f, order=(p, q), exog=np.array(xreg_train))
    model = mod.fit(maxiter=1000, disp=False)
    xreg_test = x_test.filter(like='n_')
    xreg_test = xreg_test[xreg_cols]

    fitted = model.forecast(rem_data, exog=xreg_test)
    fitted_f = sc.inverse_transform(fitted)
    fitted = fitted_f[:, 0]
    confint2 = np.array(0)
    for i in range(1, rem_data + 1):
        conf = model.get_prediction((len(trainV) + (i - 1)), exog=xreg_test[:i])
        confint = conf.conf_int(alpha=0.05)
        c1 = sc.inverse_transform([confint[0][0:x_train.shape[1]]])[0][0]
        c2 = sc.inverse_transform([confint[0][x_train.shape[1]::]])[0][0]
        confint1 = np.vstack((c1, c2)).T
        confint2 = np.append(confint2, confint1)
    confint_f = confint2[1::].reshape(rem_data, 2)
    return model, fitted, confint_f


def varmax_model_val(trainV, x_test, rem_data):
    xregressor = trainV.filter(like='n_')
    xreg_cols = xregressor.columns
    cols = [col for col in trainV.columns if col not in xregressor.columns]
    x_train = trainV[cols]
    sc = StandardScaler()
    x_train_f = sc.fit_transform(x_train)
    xreg_train = xregressor[xreg_cols]

    mat_size = 2

    aic = pd.DataFrame(np.zeros((mat_size, mat_size), dtype=float))
    bic = pd.DataFrame(np.zeros((mat_size, mat_size), dtype=float))

    # VARMAX Model
    for p in range(mat_size):
        for q in range(mat_size):
            if p == 0 and q == 0:
                continue

            # Estimate the model with no missing datapoints
            mod = VARMAX(x_train_f, order=(p, q), exog=np.array(xreg_train))
            try:
                model = mod.fit(maxiter=1000, disp=False)
                aic.iloc[p, q] = model.aic
                bic.iloc[p, q] = model.bic
            except:
                aic.iloc[p, q] = np.nan
                bic.iloc[p, q] = np.nan

    aic.iloc[0, 0] = np.nan
    bic.iloc[0, 0] = np.nan


    q = aic.min().idxmin()
    p = aic.idxmin()[q]

    mod = VARMAX(x_train_f, order=(p, q), exog=np.array(xreg_train))
    model = mod.fit(maxiter=1000, disp=False)
    xreg_test = x_test.filter(like='n_')
    xreg_test = xreg_test[xreg_cols]

    fitted = model.forecast(rem_data, exog=xreg_test)
    fitted_f = sc.inverse_transform(fitted)
    fitted = fitted_f[:, 0]
    confint2 = np.array(0)
    for i in range(1, rem_data + 1):
        conf = model.get_prediction((len(trainV) + (i - 1)), exog=xreg_test[:i])
        confint = conf.conf_int(alpha=0.05)
        c1 = sc.inverse_transform([confint[0][0:x_train.shape[1]]])[0][0]
        c2 = sc.inverse_transform([confint[0][x_train.shape[1]::]])[0][0]
        confint1 = np.vstack((c1, c2)).T
        confint2 = np.append(confint2, confint1)
    confint_f = confint2[1::].reshape(rem_data, 2)
    return model, fitted, confint_f


############ Moving Average Functions #############
def moving_average(y_train, x_test, rem_data):
    bench_df = y_train.tail(3)
    avg = bench_df.mean()
    fitted = np.repeat(avg, x_test.shape[0])
    confint = np.tile(np.array([np.min(bench_df), np.max(bench_df)]), x_test.shape[0]).reshape(rem_data, 2)
    model = 'avg'
    return model, fitted, confint


######################### Select the best model #############################
def run_best_model(x_train, y_train, x_val, y_val, best_model_name, model_df, rem_data, var, dataV):
    if best_model_name == 'rf':
        fit, fitted, fimp_sorted, confint = random_forest(x_train, y_train, x_val)
    elif best_model_name == 'varmax':
        fit, fitted, confint = varmax_model_val(dataV, x_val, rem_data)
    elif best_model_name == 'model_arima_reg':
        fit, fitted, confint = run_arima(x_train, y_train, x_val, rem_data, var)
    elif best_model_name == 'model_arima_null':
        fit, fitted, confint = run_arima_null(x_train, y_train, x_val, rem_data, var)
    elif best_model_name == 'moving_average':
        fit, fitted, confint= moving_average(y_train, x_val, rem_data)
    return fit, fitted, confint


def best_fit(x_train, y_train, x_val, y_val, model_list, rem_data, var, dataV):
    model_list.sort(key=lambda tup: tup[1])
    best_model_name = model_list[0][0]
    model_df = pd.DataFrame(model_list, columns=['model_name', 'mape', 'model'])
    mape_diff = model_df['mape'].diff().fillna(0)
    model_df['mape_diff'] = mape_diff
    if best_model_name == 'moving_average':
        ind = np.where(model_df.model_name == 'moving_average')[0][0]+1
        if mape_diff.iloc[ind] <= 10:
            best_model_name = model_list[ind][0]
    fit, fitted, confint = run_best_model(x_train, y_train, x_val, y_val, best_model_name, model_df, rem_data, var, dataV)
    return fit, fitted, best_model_name, confint


def make_confint_df(confint, y_test):
    pred_confint = pd.DataFrame()
    pred_confint['MonthYear'] = y_test['MonthYear'].copy()
    pred_confint['lower_bound'] = list(confint[:, 0])
    pred_confint['upper_bound'] = list(confint[:, 1])
    return pred_confint


############# Forecast Plot Functons ########################
def plot_fc_test_train(y_train, y_test, y_predicted, y_unseen, pred_confint, un_confint):
    trace0 = go.Scatter(x=y_train['MonthYear'], y=y_train["total_payments"], mode='lines',
                        name="Train", marker=dict(color='blue', line=dict(width=1)))
    trace1 = go.Scatter(x=y_test['MonthYear'], y=y_test['total_payments'], mode='lines',
                        name="Val Actual", marker=dict(color='red', line=dict(width=1)))
    trace2 = go.Scatter(x=y_predicted['MonthYear'], y=y_predicted['total_payments'], mode='lines',
                        name="Val Predicted", marker=dict(color='green', line=dict(width=1)))
    trace3 = go.Scatter(x=pred_confint['MonthYear'], y=pred_confint['upper_bound'], mode='none', fill='tonexty',
                        fillcolor='rgba(131, 90, 241,0.15)', showlegend=False, marker=dict(line=dict(width=1)))
    trace4 = go.Scatter(x=pred_confint['MonthYear'], y=pred_confint['lower_bound'], mode='none', fill='tonexty',
                        fillcolor='rgba(131, 90, 241,0.15)', showlegend=False, marker=dict(line=dict(width=1)))
    # trace5 = go.Scatter(x=y_unseen['MonthYear'], y=y_unseen['total_payments'], mode='lines',
    #                     name="Unseen Data", marker=dict(color='purple',line=dict(width=1)))
    # trace6 = go.Scatter(x=un_confint['MonthYear'], y=un_confint['upper_bound'], mode='none', fill='tonexty',
    #                     fillcolor='rgba(131, 90, 241,0.15)', showlegend=False, marker=dict(line=dict(width=1)))
    # trace7 = go.Scatter(x=un_confint['MonthYear'], y=un_confint['lower_bound'], mode='none', fill='tonexty',
    #                     fillcolor='rgba(131, 90, 241,0.15)', showlegend=False, marker=dict(line=dict(width=1)))

    data = [trace0, trace1, trace2, trace3, trace4]
    layout = go.Layout(go.Layout(yaxis=dict(title="Total Payments", zeroline=False),
                                 xaxis=go.layout.XAxis(
                                     title="Month-Year",
                                 ), boxmode='group', showlegend=True))
    plot_html = plotly.offline.plot({"data": data, "layout": layout}, output_type='div')
    return plot_html


def make_unseen_predictions(be_name, df, best_model_name, un_x_test, n_periods_test, var, y_test, model_list, rem_data,
                            dataV):
    # ReTrain model for unseen data
    un_x_train = df.iloc[:, 1:]
    un_y_train = df.iloc[:, 0]

    if best_model_name == 'rf':
        model, fitted, fimp_sorted, new_confint = random_forest(un_x_train, un_y_train, un_x_test)  # model for unseen data
    elif best_model_name == 'model_arima_reg':
        x_test = un_x_test[un_x_train.columns]
        model, fitted, new_confint = run_arima(un_x_train, un_y_train, x_test, n_periods_test, var)
    elif best_model_name == 'model_arima_null':
        x_test = un_x_test[un_x_train.columns]
        model, fitted, new_confint = run_arima_null(un_x_train, un_y_train, x_test, n_periods_test, var)
    elif best_model_name == 'varmax':
        model, fitted, new_confint = varmax_model_val(dataV, un_x_test, rem_data)
    elif best_model_name == 'moving_average':
        model, fitted, new_confint = moving_average(un_y_train, un_x_test, rem_data)
    # make dataframe for unseen predictions
    y_unseen = pd.DataFrame()
    y_unseen['MonthYear'] = y_test['MonthYear'].apply(lambda r: r + relativedelta(months=3))
    y_unseen['total_payments'] = list(fitted)
    return y_unseen, new_confint


# transform data for model plots
def plot_transform(y_train, y_test, y_predicted, y_unseen, pred_confint, un_confint):
    last_train = y_train.tail(1)
    last_test = y_test.tail(1)
    y_test = y_test.append(last_train)
    y_test.sort_values(by='MonthYear', inplace=True)
    y_predicted = y_predicted.append(last_train)
    y_predicted.sort_values(by='MonthYear', inplace=True)
    pred_confint = pred_confint.append(pd.DataFrame([y_train.iloc[-1, 0],
                                                     y_train.iloc[-1, 1],
                                                     y_train.iloc[-1, 1]], index=pred_confint.columns).transpose())
    pred_confint[['lower_bound', 'upper_bound']] = pred_confint[['lower_bound', 'upper_bound']].astype(float)
    pred_confint.sort_values(by='MonthYear', inplace=True)
    pred_confint['MonthYear'] = pd.to_datetime(pred_confint['MonthYear'])
    # append last item of test in confidence interval of Unseen Data
    y_unseen = y_unseen.append(last_test)
    y_unseen.sort_values(by='MonthYear', inplace=True)
    un_confint = un_confint.append(pred_confint.tail(1))
    un_confint.sort_values(by='MonthYear', inplace=True)
    un_confint['MonthYear'] = pd.to_datetime(un_confint['MonthYear'])
    un_confint[['lower_bound', 'upper_bound']] = un_confint[['lower_bound', 'upper_bound']].astype(float)

    # make plots
    x_var = 'MonthYear'
    y_var = 'total_payments'
    plot_html = plot_fc_test_train(y_train, y_test, y_predicted, y_unseen, pred_confint, un_confint)
    return plot_html


# parent function for above functions
def make_plot(be_name, var, y_train1, y_val, df_unseen, fitted, df, n_periods, rem_data, date, best_model_name,
              model_list, dataV, test_confint):
    ### Assign variables for forecast plots
    # prepare data for plot
    y_train = y_train1.reset_index()
    # Test Actual data
    y_test = y_val.reset_index()
    # Test Predicted
    y_predicted = pd.DataFrame()
    y_predicted['MonthYear'] = y_test['MonthYear'].copy()
    y_predicted['total_payments'] = list(fitted)
    # use VAR model for x_test predictions. This is required for unseen data predictions
    date2 = date + relativedelta(months=3)
    new_df = df.append(df_unseen)  # actuals+unseen
    new_df.fillna(0, inplace=True)
    un_x_test, var_flag, var_mape = create_test(new_df, df, date2, n_periods, rem_data)

    # calculate Mape on test set
    test_mape = mape(y_test['total_payments'], y_predicted['total_payments'])
    print("Test Mape",test_mape)
    print("y test",y_test['total_payments'])
    print("y pred", y_predicted['total_payments'])
    # make dataframe for unseen predictions

    y_unseen, new_confint = make_unseen_predictions(be_name, df, best_model_name, un_x_test, 3,var, y_test, model_list, rem_data, dataV)
    # calculate confidence Interval for Y_predicted
    pred_confint = make_confint_df(test_confint, y_test)
    # calculate confidence Interval for Y_Unseen
    un_confint = make_confint_df(new_confint, y_unseen)
    # Make plot
    plot_html = plot_transform(y_train, y_test, y_predicted, y_unseen, pred_confint, un_confint)
    return y_unseen, un_confint, plot_html, test_mape














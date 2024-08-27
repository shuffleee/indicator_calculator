# -*- coding: utf-8 -*-
# @Time    : 2024/8/27 21:40
# @Author  : lizhaoyang
# @File    : indicator_calculator.py
# @Desc    : 获取过去10年数据并计算windA指数日收益率、中证1000指数日收益率，组合收益率、组合净值、相对上涨比例指标、市值加权涨跌离散度等指标
# @Software: PyCharm

import pandas as pd
import sys
import time

from my_code.database.connect_wind import ConnectDatabase

# 添加当前路径到环境变量
sys.path.append('F:\quantchina\my_code')
sys.path.append('/nas92/quanyi4g/temporary/lizhaoyang/my_code/')

# 获取wind全A指数日行情、中证1000指数日行情，计算组合收益率、组合净值
class GetStockData(ConnectDatabase):
    """
    获取股票数据，包括指数数据、成分股数据、市值数据等，并计算相应的指标。
    """
    def __init__(self, start_date, end_date, index_code = '000985.CSI', windA_index_code = '8841388.WI', china1000_index_code = '000852.SH'):
        self.start_date = start_date
        self.end_date = end_date
        self.index_code = index_code
        self.windA_index_code = windA_index_code
        self.china1000_index_code = china1000_index_code

        self.date_list = None

        # 缓存查询结果，避免重复查询
        self.index_data, self.date_list = self.get_index_data(self.index_code, self.start_date, self.end_date)
        self.index_member_stock_df = self.get_index_member_stock_df(self.index_code, self.date_list)
        self.windA_index_data = self.get_windA_index_data(self.windA_code, self.start_date, self.end_date)
        self.china1000_index_data = self.get_china1000_index_data(self.china1000_code, self.start_date, self.end_date)

        # 默认初始化为 None，可以在需要时加载
        self.market_value_data = None
        self.all_stocks_daily_data = None
        self.return_volatility = None

    def execute_sql(self, sql):
        """
        执行SQL查询并从数据库中获取数据。

        :param sql: str, 要执行的SQL查询语句。
        :return: DataFrame, 查询结果的数据。
        """
        return ConnectDatabase(sql).get_data()

    def data_preprocess(self, data):
        """
        对数据进行预处理，包括日期格式转换和列类型转换。

        :param data: DataFrame, 原始数据。
        :return: DataFrame, 预处理后的数据。
        """
        data['TRADE_DT'] = pd.to_datetime(data['TRADE_DT'], format='%Y%m%d')
        # 其他需要计算的列转换成float
        other_columns = data.columns[~data.columns.isin(['S_INFO_WINDCODE', 'TRADE_DT'])]
        data[other_columns] = data[other_columns].astype(float)
        data.rename(columns={'TRADE_DT': 'date'}, inplace=True)
        data.sort_values(by=['date', 'S_INFO_WINDCODE'], inplace=True)

        return data

    # 波动率计算（N日之前 过去 20 日的收益率标准差）
    def cal_volatility(self, data, window):
        """
        计算波动率，即N日之前过去20日的收益率标准差。

        :param data: DataFrame, 收益率数据。
        :param window: int, 滚动窗口大小，用于计算波动率。
        :return: DataFrame, 计算后的波动率数据。
        """
        data = data.rolling(window=window).std()
        return data

    # 获取每日成分股list
    def get_index_member_stock_list(self, info, day):
        """
        获取某个日期的指数成分股列表。

        :param info: DataFrame, 包含成分股信息的数据。
        :param day: str, 日期，格式为'YYYYMMDD'。
        :return: list, 成分股代码列表。
        """
        info = info[(info['S_CON_INDATE'] <= day) & ((info['S_CON_OUTDATE'] >= day) | (info['S_CON_OUTDATE'].isnull()))]
        list = info['S_CON_WINDCODE'].tolist()
        return list

    # =====================================查表===============================================================
    # 查询中证全指指数以及日行情
    def get_index_data(self):
        """
        查询中证全指指数及其日行情数据。

        :return: DataFrame, 中证全指指数及其日行情数据。
        """
        sql = f'''
                SELECT TRADE_DT, S_INFO_WINDCODE, S_DQ_CLOSE,(S_DQ_CLOSE / S_DQ_PRECLOSE - 1) as index_daily_return
                FROM AINDEXEODPRICES
                WHERE S_INFO_WINDCODE = '{self.index_code}'
                AND TRADE_DT >= '{self.start_date}'
                AND TRADE_DT <= '{self.end_date}'
                '''
        index_data = self.execute_sql(sql)

        index_data = self.data_preprocess(index_data)
        self.date_list = index_data['date'].unique().tolist()

        return index_data

    # 查询wind全A指数日行情
    def get_windA_index_data(self):
        """
        查询Wind全A指数的日行情数据。

        :return: DataFrame, Wind全A指数的日行情数据。
        """
        sql = f'''
                SELECT TRADE_DT, S_INFO_WINDCODE, (S_DQ_CLOSE / S_DQ_PRECLOSE - 1) as windA_daily_return
                FROM AINDEXWINDINDUSTRIESEOD
                WHERE S_INFO_WINDCODE = '{self.windA_index_code}'
                AND TRADE_DT >= '{self.start_date}'
                AND TRADE_DT <= '{self.end_date}'
                '''
        windA_data = self.execute_sql(sql)

        windA_data = self.data_preprocess(windA_data)
        windA_data.rename(columns={'close': 'windA_close'}, inplace=True)
        windA_data['net_value'] = (1 + windA_data['windA_daily_return']).cumprod()

        return windA_data

    # 查询中证1000指数日行情
    def get_china1000_index_data(self):
        """
        查询中证1000指数的日行情数据。

        :return: DataFrame, 中证1000指数的日行情数据。
        """
        sql = f'''
                SELECT TRADE_DT, S_INFO_WINDCODE, (S_DQ_CLOSE / S_DQ_PRECLOSE - 1) as china1000_index_daily_return
                FROM AINDEXEODPRICES
                WHERE S_INFO_WINDCODE = '{self.china1000_index_code}'
                AND TRADE_DT >= '{self.start_date}'
                AND TRADE_DT <= '{self.end_date}'
                '''
        china1000_index_data = self.execute_sql(sql)

        china1000_index_data = self.data_preprocess(china1000_index_data)
        china1000_index_data['net_value'] = (1 + china1000_index_data['china1000_index_daily_return']).cumprod()

        return china1000_index_data

    def cal_combination_profit(self, windA_index_data, china1000_index_data, up_date=False):
        """
        构造一个标的（wind 全A - 中证1000)的收益率、复利的净值

         :param windA_index_data: wind 全A指数日收益率
         :param china1000_index_data: 中证1000指数日收益率
         :param up_date: 是否更新数据

         :return: DataFrame, 组合的收益率和净值数据。
        """
        combination_data = pd.merge(windA_index_data, china1000_index_data, on='date', how='inner')
        combination_data['combination_return'] = (combination_data['windA_daily_return'] - combination_data['china1000_index_daily_return'])

        if not up_date:
            # 前一天的 combination_net_value * （1 + 今天的combination_return）
            combination_data['combination_net_value'] = (1 + combination_data['combination_return']).cumprod()
            # 保留收益率、复利的净值
            combination_data = combination_data[['date', 'windA_daily_return', 'china1000_index_daily_return', 'combination_return','combination_net_value']]

        else:  # 更新数据
            pre_data = pd.read_csv('combination_data.csv')

            pre_com_net_value = pre_data[['date', 'combination_net_value']].copy()
            pre_com_net_value['date'] = pd.to_datetime(pre_com_net_value['date'])
            combination_data['date'] = pd.to_datetime(combination_data['date'])

            # 对齐日期并获取历史数据对应的 combination_net_value
            combination_data = pd.merge(combination_data, pre_com_net_value, on='date', how='left')
            # 更新最后一天的净值
            combination_data.loc[combination_data.index[-1], 'combination_net_value'] = (combination_data.loc[combination_data.index[-2], 'combination_net_value'] *
                                                                                         (1 + combination_data.loc[combination_data.index[-1], 'combination_return']))
            combination_data = combination_data[['date', 'windA_daily_return', 'china1000_index_daily_return', 'combination_return','combination_net_value']]

        return combination_data

    def get_index_member_stock_df(self):
        """
        获取某个指数在特定日期的成分股信息，并展开为包含日期的DataFrame。

        :return: DataFrame, 包含日期和成分股代码的数据。
        """
        sql = f'''
                 SELECT S_INFO_WINDCODE, S_CON_WINDCODE, S_CON_INDATE, S_CON_OUTDATE
                 FROM AINDEXMEMBERS
                 WHERE S_INFO_WINDCODE = '{self.index_code}'
                '''
        index_member_info = self.execute_sql(sql)

        index_member_expanded = []
        for day in self.date_list:
            day_str = day.strftime('%Y%m%d')

            index_member_stock_list = self.get_index_member_stock_list(index_member_info, day_str)
            temp_df = pd.DataFrame({'date': day_str, 'S_INFO_WINDCODE': index_member_stock_list})

            index_member_expanded.append(temp_df)

        # 将成分股信息展开为一个包含日期的 DataFrame
        index_member_expanded_df = pd.concat(index_member_expanded, ignore_index=True)
        index_member_expanded_df['date'] = pd.to_datetime(index_member_expanded_df['date'], format='%Y%m%d')

        return index_member_expanded_df

    # 查询中证全指成分股的日行情
    def get_index_member_data(self, index_member_stock_df, all_stocks_data): # 此处传进来all_stocks_data是因为读历史数据在外面
        """
        查询中证全指成分股的日行情并计算波动率。

        :param index_member_stock_df: DataFrame, 成分股信息数据。
        :param all_stocks_data: DataFrame, 所有股票的日行情数据。
        :return: DataFrame, 包含波动率和日收益率的成分股数据。
        """
        start_time = time.time()
        # 获取每日成分股的日行情(只查询一次）
        sql = f'''
                        SELECT TRADE_DT, S_INFO_WINDCODE,(S_DQ_ADJCLOSE / S_DQ_ADJPRECLOSE  - 1) as ADJ_daily_return
                        FROM ASHAREEODPRICES
                        WHERE TRADE_DT >= '{self.start_date}' AND TRADE_DT <= '{self.end_date}'
                    '''
        all_stocks_data = self.execute_sql(sql)
        print('查个股每日行情表所花时间为{}秒'.format(time.time() - start_time))

        # 去除第一列(读取csv文件时会多出一列)
        # all_stocks_daily_data = all_stocks_daily_data.iloc[:,1:]

        all_stocks_data = all_stocks_data[~all_stocks_data['S_INFO_WINDCODE'].str.contains('BJ')]
        all_stocks_data = self.data_preprocess(all_stocks_data)

        # 合并每日成分股和每日行情数据
        all_stocks_merged_data = pd.merge(index_member_stock_df, all_stocks_data, on=['date', 'S_INFO_WINDCODE'],
                                   how='left')
        all_stocks_merged_data = all_stocks_merged_data[~all_stocks_merged_data['S_INFO_WINDCODE'].str.contains('BJ')]
        all_stocks_merged_data.sort_values(by=['date', 'S_INFO_WINDCODE'], inplace=True)

        # 向量化(此处可能不需要用向量化计算，如 689009.SH 这只股票在20211213前没有数据，会导致整列数据前面日期都为NaN)
        all_stocks_data_vectorization = all_stocks_merged_data.pivot(index='date', columns='S_INFO_WINDCODE',
                                                              values='ADJ_daily_return')
        ADJ_daily_return_vectorization = all_stocks_merged_data.pivot(index='date', columns='S_INFO_WINDCODE',
                                                               values='ADJ_daily_return')
        
        self.return_volatility = self.cal_volatility(ADJ_daily_return_vectorization, 20)
        self.return_volatility = self.return_volatility.loc[all_stocks_data_vectorization.index, all_stocks_data_vectorization.columns]

        return all_stocks_merged_data

    # 计算相对上涨比例指标
    def cal_relative_rise_ratio(self, index_data, stocks_data):
        """
        计算相对上涨比例指标，即在中证全指的成分股中，当日上涨幅度大于指数收益率的股票比例。

        :param index_data: DataFrame, 包含指数日行情数据。
        :param stocks_data: DataFrame, 包含所有成分股的日收益率数据。
        :return: DataFrame, 包含计算后的相对上涨比例指标的数据。
        """
        # 按照每日计算相对上涨比例指标，中证全指里面当日大于指数收益率成分股数量占比(是否向量化去计算要更快一点)
        for date in self.date_list:
            index_daily_return = float(index_data[index_data['date'] == date]['index_daily_return'].values[0])
            stock_daily_return = stocks_data[stocks_data['date'] == date]['ADJ_daily_return'].fillna(0).astype(float) # 将 decimal.Decimal 转换为 float

            rise_ratio = (stock_daily_return > index_daily_return).sum() / len(stock_daily_return)

            index_data.loc[index_data['date'] == date, 'relative_rise_ratio'] = rise_ratio
            # 全市场数量（验证指标）
            index_data.loc[index_data['date'] == date, 'all_stock_num'] = len(stock_daily_return)
            # 比指数高的个股数量
            index_data.loc[index_data['date'] == date, 'rise_stock_num'] = (
                        stock_daily_return > index_daily_return).sum()

        index_data = index_data[['date', 'all_stock_num', 'rise_stock_num', 'relative_rise_ratio']]
        print("相对上涨比例指标：\n", index_data)

        return index_data

    # 市值加权涨跌离散度
    def get_market_value(self, market_value_data):
        """
        计算市值加权涨跌离散度。即基于市值加权的日波动率计算。

        :param market_value_data: DataFrame, 包含股票的市值数据。
        :return: DataFrame, 包含市值加权涨跌离散度和其他统计数据。
        """
        start_time = time.time()
        # 获取市值数据
        sql = f'''
                SELECT S_INFO_WINDCODE, TRADE_DT, S_VAL_MV
                FROM ASHAREEODDERIVATIVEINDICATOR
                WHERE TRADE_DT >= '{self.start_date}' AND TRADE_DT <= '{self.end_date}'
                '''
        market_value_data = self.execute_sql(sql)
        print('查市值表所花时间为{}秒'.format(time.time() - start_time))
        # market_value_data = pd.read_csv('market_value_data.csv')

        # 剔除后缀名BJ的股票
        market_value_data = market_value_data[~market_value_data['S_INFO_WINDCODE'].str.contains('BJ')]
        market_value_data = self.data_preprocess(market_value_data)

        # 市值权重
        market_value_data = market_value_data.pivot(index='date', columns='S_INFO_WINDCODE', values='S_VAL_MV')  # 向量化
        market_value_data = market_value_data.fillna(0)

        # (验证指标）每日波动率均值，注意有的股票没上市向量化后也会在较早日期出现NaN，在填充0之前
        volatility_mean = self.return_volatility.mean(axis=1, skipna=True)  # skipna=True 忽略NaN
        volatility_median = self.return_volatility.median(axis=1, skipna=True)  # 中位数

        # 计算市值加权涨跌离散度 (∑ mv * vol) / ∑ mv
        self.return_volatility = self.return_volatility.fillna(0)
        # 删除为0的行
        return_volatility = return_volatility[(return_volatility != 0).any(axis = 1)]
        aligned_market_value_data = market_value_data.loc[self.return_volatility.index, self.return_volatility.columns].astype(float)

        weighted_volatility = self.return_volatility * aligned_market_value_data

        sum_weighted_volatility = weighted_volatility.sum(axis=1, skipna=True)
        sum_market_value = aligned_market_value_data.sum(axis=1, skipna=True)

        rise_and_fall_dispersion = sum_weighted_volatility / sum_market_value
        rise_and_fall_dispersion_df = pd.DataFrame(rise_and_fall_dispersion, columns=['rise_and_fall_dispersion'])

        # 市值加权均值
        rise_and_fall_dispersion_df['volatility_mean'] = volatility_mean
        rise_and_fall_dispersion_df['volatility_median'] = volatility_median
        print("市值加权涨跌离散度：\n", rise_and_fall_dispersion_df)

        return rise_and_fall_dispersion_df

    # def get_N_tradeday(self, now_data, n):

    def data_update(self):
        """
        更新数据，包括计算组合净值、相对上涨比例和市值加权涨跌离散度，并将结果追加到CSV文件中。
        """
        now_date = time.strftime('%Y%m%d', time.localtime())
        self.end_date = now_date
        # 往前取20个交易日
        # pre_date = self.get_tradeday(now_data, 20)
        pre_date = '20240726'
        self.start_date = pre_date

        new_data = self.go()
        # 取出最新的数据
        new_data = new_data[new_data['date'] == self.end_date]

        with open('combination_data.csv', 'a') as f:
            new_data.to_csv(f, header=False, index=False)

    def go(self):
        """
        获取数据并计算指标。

        :return: DataFrame, 包含市值加权涨跌离散度和其他统计数据。
        """
        windA_index_data = self.get_windA_index_data(self)
        china1000_index_data = self.get_china1000_index_data(self)
        # 组合净值涉及到累积计算，所以需要取出所有数据
        combination_profit = self.cal_combination_profit(windA_index_data, china1000_index_data, up_date=True)

        # 获取中证全指日行情数据
        index_data = self.get_index_data(self)
        # 计算相对上涨比例指标
        index_member_stock_df = self.get_index_member_stock_df(self)
        all_stock_data = self.get_index_member_data(self, index_member_stock_df, self.all_stocks_daily_data)
        relative_rise_ratio = self.cal_relative_rise_ratio(index_data, all_stock_data)
        combination_relative_rise_ratio = pd.merge(combination_profit, relative_rise_ratio, on='date',
                                                    how='inner')
        # 计算市值加权涨跌离散度
        return_volatility_df = self.get_market_value(self, self.market_value_data)
        combination_data = pd.merge(combination_relative_rise_ratio, return_volatility_df, on='date',
                                    how='inner')
        
        return combination_data

if __name__ == '__main__':
    
    stock_data = GetStockData(start_date='20140825', end_date='20240823')

    combination_data = stock_data.go()

    # # 读取数据
    # all_stocks_daily_data = pd.read_csv('all_stocks_daily_data.csv')
    # # 读取市值数据
    # market_value_data = pd.read_csv('market_value_data.csv')

    combination_data.to_csv('combination_data.csv', index=False)

    # 如果需要更新数据，调用data_update方法
    stock_data.data_update()
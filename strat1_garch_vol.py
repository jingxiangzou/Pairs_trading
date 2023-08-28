import pandas as pd
import numpy as np
import scipy as sp
import copy
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
import arch
import copy


# 这个就是正常版本的，PNL除500的名义本金
class PairTrading:
    def __init__(self, df1, df2, df_beta, df_multiplier, window_list1, window_list2, open_list1, open_list2,
                 gain_list, loss_list, base_amount, start, mean_reverts):
        self.df1 = copy.deepcopy(df1)  # 500 ajdusted close
        self.df2 = copy.deepcopy(df2)  # 1000 adj close
        self.df_beta = copy.deepcopy(
            df_beta)  # 500 dui 1000 reg get beta # no idea where beta comes from and how it is calculated
        self.df_multiplier = copy.deepcopy(df_multiplier)  # multipier between 500 and 1000
        self.window_list1 = window_list1  # windows of return calculation
        self.window_list2 = window_list2  # windows of return calculation in 15-16 when the strategy fails
        self.open_list1 = open_list1  # threshold for opeing accounts : 1% 2% 3%
        self.open_list2 = open_list2  # threshold condition for validity of the strategy when the long term negative return is not too big : -8% -10%
        self.gain_list = gain_list
        self.loss_list = loss_list
        self.mrvs = mean_reverts
        self.base_amount = base_amount  # amount of position
        self.start = start
        self.combine_backtesting()

    def combine_backtesting(self):  # this func reports the execution results of the pair_trading_backtesting
        # per different values for the parameters
        #
        total_record_col = ["回归线", "短线window", "短线开仓信号（500-1000收益率）", "长线window",
                            "长线filter信号（500-1000收益率）",
                            "止盈线", "止损线", "总开仓次数",
                            "赚钱次数", "赚钱比例(%)",
                            "平均持仓时间", "赚钱时平均持仓时间", "亏钱时平均持仓时间", "总盈亏百分比",
                            "总盈亏（绝对数额）", "基础手数", "夏普"]
        df_record_final = pd.DataFrame(columns=total_record_col)
        for window1 in self.window_list1:
            for window2 in self.window_list2:
                for open1 in self.open_list1:
                    for open2 in self.open_list2:
                        for stop_gain in self.gain_list:
                            for stop_loss in self.loss_list:
                                for mrv in self.mrvs:
                                    df_combine, df_record, info = \
                                        self.pair_trading_backtesting(window1, window2, open1, open2, stop_gain, stop_loss, mrv)
                                    df_line = pd.DataFrame(info, index=[0])
                                    df_record_final = pd.concat([df_record_final, df_line])
        df_record_final.to_csv("result_garch/Combined Final Result with amount of CSI500={} {}.csv".
                               format(self.base_amount, self.start), encoding="gbk", index=False)

    # when the parameters are set, the pair trading's backtesting takes place here
    def pair_trading_backtesting(self, window1, window2, open1, open2, stop_gain, stop_loss, mrv):

        cal_rolling_mean(self.df1, window1, window2)  # 对中证500 算了两个不同window的跨期收益率 window_short & window_long
        cal_rolling_mean(self.df2, window1, window2)  # 对中证1000 算了两个不同window的跨期收益率 window_short & window_long
        get_beta(self.df1, self.df_beta, self.df_multiplier, self.start)  # 获取了该月份对应的Beta和multiplier

        # monthly avg price CSI500 / CI1000 is multiplier
        # df_beta is the return beta of CSI500 regressed on CSI1000
        # so to hedge, we need to hedge w.r.t. the abs price
        # R500 = multiplier * R1000
        # S500 = beta * S1000
        # Delta500 = R500 * S500 = beta * multiplier * Delta1000

        df_combine = pd.concat([self.df1, self.df2], axis=1).dropna()  # horizontally
        print(df_combine.head())  # flag
        print(df_combine.columns)
        df_combine.columns = ["收盘价1", "rolling_mean1_short", "rolling_mean1_long", "Beta", "Multiplier", "收盘价2",
                              "rolling_mean2_short", "rolling_mean2_long"]  # 共 8 列
        print(df_combine.head())  # flag
        print(df_combine.columns)
        # 以下将CSI500视为资产而CSI1000视为市场的代表
        # 不同window使用同一个beta

        df_combine["rolling_difference_short"] = df_combine["rolling_mean1_short"] - df_combine["Beta"] * df_combine[
            "rolling_mean2_short"]  # window1
        df_combine["rolling_difference_long"] = df_combine["rolling_mean1_long"] - df_combine["Beta"] * df_combine[
            "rolling_mean2_long"]  # window2

        # rolling std as estimator for sigma
        sig1 = df_combine["rolling_difference_short"].std()
        sig2 = df_combine["rolling_difference_long"].std()

        # apply garch to the time series which is the spread created by hedge
        # df_combine.to_csv('df_combine_t2.csv')
        date = df_combine.index
        # for short windows there are window1 NANs in the head
        model1 = arch.arch_model(df_combine["rolling_difference_short"], vol='GARCH', p=1, q=1)
        rlt1 = model1.fit()
        mug1 = rlt1.params[0]
        omega1 = rlt1.params[1]
        alpha1 = rlt1.params[2]
        beta1 = rlt1.params[3]
        df_combine['ht1'] = pd.Series([0] * (len(date)))
        df_combine.loc[date[0], 'ht1'] = df_combine["rolling_difference_short"].std()
        for index in range(1, len(date)):
            df_combine.loc[date[index], "ht1"] \
                = omega1 + alpha1 * ((df_combine.loc[date[index - 1], "rolling_difference_short"] - mug1) ** 2) \
                  + beta1 * (df_combine.loc[date[index - 1], 'ht1'])
        # df_combine.to_csv('test2.csv')

        model2 = arch.arch_model(df_combine["rolling_difference_long"], vol='GARCH', p=1, q=1)
        rlt2 = model2.fit()
        mug2 = rlt2.params[0]
        omega2 = rlt2.params[1]
        alpha2 = rlt2.params[2]
        beta2 = rlt2.params[3]
        df_combine['ht2'] = pd.Series([0] * (len(date)))
        df_combine.loc[date[0], 'ht2'] = df_combine["rolling_difference_long"].std()
        for index in range(1, len(date)):
            df_combine.loc[date[index], "ht2"] \
                = omega2 + alpha2 * ((df_combine.loc[date[index - 1], "rolling_difference_long"] - mug2) ** 2) \
                  + beta2 * (df_combine.loc[date[index - 1], 'ht2'])

        df_combine['cd1'], df_combine['cd2'], df_combine['cd3'] = pd.Series([False] * (len(date))), \
                                                                  pd.Series([False] * (len(date))), \
                                                                  pd.Series([False] * (len(date)))

        for index in range(0, len(date)):
            df_combine.loc[date[index], 'cd1'] = df_combine.loc[date[index], 'rolling_difference_short'] > \
                                                 open1 * np.sqrt(df_combine.loc[date[index], 'ht1'])
            df_combine.loc[date[index], 'cd2'] = df_combine.loc[date[index], 'rolling_difference_long'] > \
                                                 open2 * np.sqrt(df_combine.loc[date[index], 'ht2'])
            df_combine.loc[date[index], 'cd3'] = df_combine.loc[date[index], 'rolling_difference_long'] < \
                                                 - open2 * np.sqrt(df_combine.loc[date[index], 'ht2'])


        # df_combine.to_csv('with sig1 and sig2_1.csv')
        df_combine["Status"] = df_combine['cd1'] & df_combine["cd2"] & df_combine["cd3"]
        # "Status"  just reports the spread spot value as relative to the shreshold
        # "Status" is what open and stop is based on

        df_combine.to_csv('test1.csv')  # wait and see

        # what is the spread created by hedge? well, it is delta500 - delta1000 * multiplier * beta
        # and we enter into the shorting of a spread when it is larger than 1 unit
        # which means we short 1 delta500 and long multiplier * beta * delta1000

        df_combine["PnL_day"] = self.base_amount * (
                (df_combine["收盘价2"] - df_combine["收盘价2"].shift(1)) * df_combine["Beta"] * df_combine[
            "Multiplier"] -
                (df_combine["收盘价1"] - df_combine["收盘价1"].shift(1)))
        df_combine = df_combine.dropna()

        record_col = ["start_date", "end_date", "duration", "total_pnl", "total_pnl_pct",
                      "total_pnl_annual"]
        # do the backtesting, find the place to start
        df_record = pd.DataFrame(columns=record_col)

        status_list = df_combine["Status"].to_list()

        print('status list\n', status_list)
        if True in status_list:
            first_open = np.where(status_list)[0][0]  # where in starts 'True' first appears
            print('first_open', np.where(status_list))
            index_start = first_open + 1  # Second day
            index = index_start
        else:
            index_start = len(df_combine)  # Second day
            index = index_start

        # rv 'index' is set to be the same as index_start, but keep in mind that it is just used in iteration
        # 'index_start' is a constant value which is the day before opening serving as the opening signal

        while index < len(df_combine) - 1:
            print("It is Index {} now".format(index))
            index1, info = self.cal_strategy(df_combine, index_start, stop_gain, stop_loss, mrv)
            print(df_record)
            df_line = pd.DataFrame(info, index=[0])
            print('df_line =', df_line.head())
            df_record = pd.concat([df_record, df_line])

            print('df_record =', df_record)
            if True in status_list[index1:]:
                index_start = np.where(status_list[index1:])[0][0] + 1 + index1
            else:
                # 提前结束
                index_start = len(df_combine) - 1
            index = index_start

        total_record_col = ["回归线", "短线window", "短线开仓信号（500-1000收益率）", "长线window",
                            "长线filter信号（500-1000收益率）",
                            "止盈线", "止损线", "总开仓次数", "赚钱次数",
                            "赚钱比例(%)", "平均持仓时间", "赚钱时平均持仓时间", "亏钱时平均持仓时间", "总盈亏百分比",
                            "总盈亏（绝对数额）", "基础手数", "夏普"]
        number_record = len(df_record)
        number_gain = len(df_record[df_record["total_pnl_pct"] > 0])
        if len(df_record) > 0:
            pct_gain = number_gain / number_record * 100
        else:
            pct_gain = 0
        duration_mean = df_record["duration"].mean()
        duration_gain = df_record.loc[df_record["total_pnl_pct"] > 0, "duration"].mean()
        duration_loss = df_record.loc[df_record["total_pnl_pct"] < 0, "duration"].mean()
        total_gain_pct = df_record["total_pnl_pct"].sum()
        total_gain = df_record["total_pnl"].sum()
        if len(df_record) > 0:
            sharpe = df_record["total_pnl_pct"].mean() / df_record["total_pnl_pct"].std()
        else:
            sharpe = 0
        info_final = dict(zip(total_record_col,
                              [mrv, window1, open1, window2, open2, stop_gain, stop_loss, number_record, number_gain,
                               pct_gain, duration_mean, duration_gain, duration_loss, total_gain_pct, total_gain,
                               self.base_amount, sharpe]))
        df_combine.to_csv(
            "result_garch/Original window1={} open1={} window2={} open2={} gain={}loss={} base={} date={} mrv={}.csv".
            format(window1, open1, window2, open2, stop_gain, stop_loss, self.base_amount, self.start, mrv),
            encoding="gbk")
        df_record.to_csv(
            "result_garch/Record window1={} open1={} window2={} open2={} gain={}loss={} base={} date={} mrv={}.csv".
            format(window1, open1, window2, open2, stop_gain, stop_loss, self.base_amount, self.start, mrv),
            encoding="gbk", index=False)
        return df_combine, df_record, info_final

    def cal_strategy(self, df_combine, index_start, stop_gain, stop_loss, mrv):
        index = index_start
        index_list = df_combine.index
        start_date = index_list[index]
        df_combine.loc[index_list[index], "Open"] = True
        df_combine.loc[index_list[index], "Event"] = "开仓"
        df_combine.loc[index_list[index], "PnL_Combine"] = \
            df_combine.loc[index_list[index_start]:index_list[index], "PnL_day"].sum()
        df_combine.loc[index_list[index], "PnL_Combine_pct"] = df_combine.loc[index_list[index], "PnL_Combine"] / \
                                                               (df_combine.loc[
                                                                    index_list[
                                                                        index_start], "收盘价1"] * self.base_amount)

        index += 1
        date = df_combine.index

        while index < len(df_combine) - 1 and stop_loss < df_combine.loc[index_list[index - 1], "PnL_Combine_pct"] \
                < stop_gain \
                and abs(df_combine.loc[date[index], 'rolling_difference_long']) > mrv \
                and abs(df_combine.loc[date[index], 'rolling_difference_short']) > mrv:
            df_combine.loc[index_list[index], "PnL_Combine"] = \
                df_combine.loc[index_list[index_start]:index_list[index], "PnL_day"].sum()
            df_combine.loc[index_list[index], "PnL_Combine_pct"] = df_combine.loc[index_list[index], "PnL_Combine"] / \
                                                                   (df_combine.loc[
                                                                        index_list[
                                                                            index_start], "收盘价1"] * self.base_amount)
            index += 1
        df_combine.loc[index_list[index], "Stop"] = True
        df_combine.loc[index_list[index], "Event"] = "平仓"
        end_date = index_list[index]
        duration = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days
        total_pnl = df_combine.loc[index_list[index - 1], "PnL_Combine"]
        total_pnl_pct = df_combine.loc[index_list[index - 1], "PnL_Combine_pct"]
        total_pnl_annual = total_pnl_pct / duration * 365
        info = dict(zip(["start_date", "end_date", "duration", "total_pnl", "total_pnl_pct", "total_pnl_annual"],
                        [start_date, end_date, duration, total_pnl, total_pnl_pct, total_pnl_annual]))
        return index, info


def cal_rolling_mean(df, window_short, window_long):  # to  give value to rolling_mean_short & rolling_mean_long
    date = df.index
    for index in range(window_short, len(df)):
        # rolling_mean_short = 某一时点相对于 window_short 天前的收益率
        df.loc[date[index], "rolling_mean_short"] = (df.loc[date[index], "收盘价"] - df.loc[
            date[index - window_short], "收盘价"]) / \
                                                    df.loc[date[index - window_short], "收盘价"]

    for index in range(window_long, len(df)):
        df.loc[date[index], "rolling_mean_long"] = (df.loc[date[index], "收盘价"] - df.loc[
            date[index - window_long], "收盘价"]) / \
                                                   df.loc[date[index - window_long], "收盘价"]
    return df


def get_beta(df, df_beta, df_multiplier, start="201811"):
    date = df.index
    for index in range(len(df)):
        d = pd.to_datetime(date[index])  # 2021-09-13 00:00:00
        d = d.strftime("%Y-%m-%d")  # 2021-09-13 00:00:00
        d = d[:4] + d[5:7]  # 202109
        if d > start:
            df.loc[date[index], "Beta"] = df_beta.loc[d[:6], "Beta"]
            # print('the month =', d)
            # print('beta =' , df_beta.loc[d[:6], "Beta"])
            df.loc[date[index], "Multiplier"] = df_multiplier.loc[int(d[:6]), "Multiplier"]
    return df  # 每月取一个beta取一个multiplier


if __name__ == "__main__":
    df_500 = pd.read_csv("data/CSI500.csv", index_col="日期", encoding="gbk")
    df_500 = df_500.iloc[:-1, :1]
    df_1000 = pd.read_csv("data/CSI1000.csv", index_col="日期", encoding="gbk")
    df_1000 = df_1000.iloc[:-1, :1]
    df_beta = pd.read_csv("data/Beta.csv", index_col="月份", encoding="gbk")
    df_multiplier = pd.read_csv("data/multiplier.csv", index_col="月份")
    # print(cal_rolling_mean(df_500, 240))
    # print(pair_trading_backtesting(df_500, df_1000,df_beta,240,0.02,0.02,-0.02,1))
    # print(get_beta(df_500, df_beta))
    window_list1 = [21, 60]  # [10, 20, 60]
    open_list1 = [0.01, 0.015, 0.02, 0.025]
    # window_list2 = [120, 180, 240, 300]
    window_list2 = [120, 240]  # [180, 240, 300]
    open_list2 = [-0.015, -0.03]  # [-1.2, -1.3]
    gain_list = [0.02]  # the final PnL is not sensitive to stop_gain
    loss_list = [-0.02]  # the final PnL is not sensitive to stop_loss
    mean_reverts = [0.0005, 0.0008, 0.001]
    # window_list = [240]
    # open_list = [0.02]
    # gain_list = [0.01, 0.02]
    # loss_list = [-0.01]

    PairTrading(df_500, df_1000, df_beta, df_multiplier, window_list1, window_list2, open_list1, open_list2,
                gain_list, loss_list, 10000, "201511", mean_reverts)
    # PairTrading(df_500, df_1000, df_beta, df_multiplier, window_list1, window_list2, open_list1, open_list2,
    #            gain_list, loss_list, 10000, "201711")
    # PairTrading(df_500, df_1000, df_beta, df_multiplier, window_list1, window_list2, open_list1, open_list2,
    #            gain_list, loss_list, 10000, "201811")
    # PairTrading(df_500, df_1000, df_beta, df_multiplier, window_list1, window_list2, open_list1, open_list2,
    #            gain_list, loss_list, 10000, "201911")
    # PairTrading(df_500, df_1000, df_beta, df_multiplier, window_list1, window_list2, open_list1, open_list2,
    #            gain_list, loss_list, 10000, "202011")
    # here the base amount stays unchanged as it is just a multiplier
    # tests are conducted per different starting dates

    # print(combine_backtesting(df_500, df_1000, df_beta, window_list, open_list, gain_list, loss_list, base_amount=10000, date="201811"))
    # print(combine_backtesting(df_500, df_1000, df_beta, window_list, open_list, gain_list, loss_list, base_amount=10000,
    #                           date="201511"))
    # print(combine_backtesting(df_500, df_1000, df_beta, window_list, open_list, gain_list, loss_list, base_amount=10000,
    #                           date="201711"))
    # print(combine_backtesting(df_500, df_1000, df_beta, window_list, open_list, gain_list, loss_list, base_amount=10000,
    #                           date="201911"))
    # print(combine_backtesting(df_500, df_1000, df_beta, window_list, open_list, gain_list, loss_list, base_amount=10000,
    #                           start="202111"))
    # print(combine_backtesting(df_500, df_1000, df_beta, window_list, open_list, gain_list, loss_list, base_amount=1, date="201811"))
    # print(combine_backtesting(df_500, df_1000, df_beta, window_list, open_list, gain_list, loss_list, base_amount=100,date="201811"))

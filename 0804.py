#coding:utf-8

import numpy as np
from datetime import datetime
import xgboost as xgb
import os, os.path
from matplotlib import pyplot
import pandas as pd
from sklearn.preprocessing import LabelEncoder
# 待添加的东西：link_width  link_length  link_class done
# 待改进: encode decode部分优化 done
# 按平均通行时间划分等级 优化  done
# 缺失值的处理

data = pd.read_csv('gy_contest_link_info.txt', sep=';')
le = LabelEncoder()
le.fit(data['link_ID'])

def maelikeobj(preds, dtrain):
    labels = dtrain.get_label()
    res = np.array(preds - labels)
    grad = (np.exp(2*res) -1) / (np.exp(2*res) +1)
    #hess = 4*np.exp(2*res) / np.power((np.exp(2*res)+1),2)
    hess = np.repeat([1],len(res))
    return grad ,hess


def eval_log_mape(preds, dtrain):
    labels = dtrain.get_label()
    # return a pair metric_name, result
    return 'mapelike' , float(np.mean(abs(np.exp(labels) - np.exp(preds))/(np.exp(labels)-1)))


class ProcessData:
    """处理类"""
    def __init__(self, weekspan, dayspan, secspan):
        self.weeks = weekspan
        self.days = dayspan
        self.secs = secspan
        self.samples = {}
        # hour 8
        self.levels = [0,6.4,15.5,32,60,200]
        self.stats = []
        self.weather = {}
 

    def Weather_Features(self):
        for line in open('Weather Features.txt'):
            terms = line.strip().split(';')
            dayid = int(terms[0])
            hourid = int(terms[1])
            SSD = float(terms[2])
            VIS = float(terms[3])
            if dayid not in self.weather:
                self.weather[dayid] = {}
            self.weather[dayid][hourid] = {'SSD':SSD,'VIS':VIS}


    def get_level(self, linkcube):
        """根据道路的历史通过时间划分等级，lincube[dayid][hourid][deltaid]"""
        tmp = []
        for day, dayvalue in linkcube.items():
            # 6月的数据不进入
            if day > 152:
                continue
            for hour,hourvalue in dayvalue.items():
                # 只考虑早上8:00~9:00的平均通行时间
                if hour != 8:
                    continue
                for delta,deltavalue in hourvalue.items():
                    tmp.append(deltavalue['traveltime'])
        stat = sum(tmp) / len(tmp)
        self.stats.append(stat)
        print "stat.....%.3f" % stat
        for idx in range(len(self.levels)-1):
            if self.levels[idx] <= stat < self.levels[idx+1]:
                return idx, stat



    def get_delta_mean(self, linkcube):
        """获取每条道路每2min间隔的历史均值"""
        tmp = {}
        for day, dayvalue in linkcube.items():
            # 6月的数据不进入
            if day > 152:
                continue
            for hour, hourvalue in dayvalue.items():
                for delta, deltavalue in hourvalue.items():
                        tmp.setdefault(hour, {})
                        tmp[hour].setdefault(delta, [])
                        tmp[hour][delta].append(deltavalue['traveltime'])
        mean_value = {}
        for hourid in range(24):
            for deltaid in range(30):
                mean_value.setdefault(hourid, {})
                mean_value[hourid].setdefault(deltaid, [])
                try:
                    mean_value[hourid][deltaid] = sum(tmp[hourid][deltaid]) / len(tmp[hourid][deltaid])
                except:
                    mean_value[hourid][deltaid] = 0.0
        return mean_value



    def get_dayid_weekdayis(self, date_info):
        """获取当天是一年中的第几天，一周中的第几天[0,6]"""
        dt_data = datetime.strptime(date_info, '%Y-%m-%d')
        # %j >> 一年中的第几天   %w >> 一周中的第几天  %U >> 一年中的第几周
        dayid, weekdayid = dt_data.strftime('%j'), dt_data.strftime('%w')
        return int(dayid), int(weekdayid)


    def get_deltaid(self, time_interval):
        """将1小时按照2分钟间隔划分成30个deltaid, deltaid [0,29]"""
        x = time_interval.strip('[').split(',')[0].split(' ')[1].split(':')[1]
        deltaid = int(x) / 2
        return deltaid


    def get_hourid(self, time_interval):
        """Hourid [0,23]"""
        hourid = time_interval.strip('[').split(',')[0].split(' ')[1].split(':')[0]
        return int(hourid)


    def get_isholiday(self, date_info):
        """0 工作日  1 周末  2 假期"""
        dayid, weekdayid = self.get_dayid_weekdayis(date_info)
        if date_info in ('2016-04-02','2016-04-03','2016-04-04','2016-04-30','2016-05-01','2016-05-02','2016-06-09','2016-06-10','2016-06-11'):
            return 2
        if date_info == '2016-06-12':
            return 0
        if weekdayid == 0 or weekdayid == 6:
            return 1
        else:
            return 0


    def slide_window(self, values):
        """滑窗平均值"""
        mean_arr, median_arr = [], []
        for i in range(len(values)-2):
            tmp = values[i:i+3]
            mean_arr.append(np.mean(tmp))
            median_arr.append(np.median(tmp))
        return (mean_arr,median_arr)


    def process_today_series(self,today_series):
        arr = []
        for i in range(len(today_series)-1,-1,-1):
            arr.append(np.mean(today_series[i:]))
        return arr


    def link_plot(self, lastlinkid, linkcube, deltaid):
        """画出link的历史值/// linkcube {dayid{hourid{deltaid}}}"""
        print "Ploting............linkid=%s" % lastlinkid
        X, Y = [], []
        for day, dayvalues in linkcube.items():
            for hour, hourvalues in dayvalues.items():
                if hour != 8:
                    continue
                for delta, deltavalues in hourvalues.items():
                    if delta != deltaid:
                        continue
                    Y.append(deltavalues['traveltime'])
                    X.append(day)
        # pyplot.plot(X,Y)
        # pyplot.xlabel("DAY")
        # pyplot.ylabel("Travel Time")
        # pyplot.title("link=%s"%lastlinkid)
        linkcube = self.checkoutliers(lastlinkid, linkcube)
        Y_2 = []
        for day, dayvalues in linkcube.items():
            for hour, hourvalues in dayvalues.items():
                if hour != 8:
                    continue
                for delta, deltavalues in hourvalues.items():
                    if delta != deltaid:
                        continue
                    Y_2.append(deltavalues['traveltime'])
        fig,ax = pyplot.subplots()
        ax.plot(X,Y,label='original data')
        ax.plot(X,Y_2,label='cleaned data')
        pyplot.legend()
        pyplot.xlabel('DAY')
        pyplot.ylabel('Travel Time')
        pyplot.show()


    def explore_holiday_trend(self, lastlinkid, linkcube):
        """画出节假日2,工作日0,周末1的数据区别"""
        print "Exploring..............linkid=%s" % lastlinkid
        Y0,Y1,Y2=[],[],[]
        for day,dayvalues in linkcube.items():
            for hour, hourvalues in dayvalues.items():
                if hour != 8:
                    continue
                for delta, deltavalues in hourvalues.items():
                    if deltavalues['isholiday'] == 0:
                        Y0.append(deltavalues['traveltime'])
                    elif deltavalues['isholiday'] == 1:
                        Y1.append(deltavalues['traveltime'])
                    else:
                        Y2.append(deltavalues['traveltime'])
        fig,ax=pyplot.subplots()
        ax.plot(Y0,label='Working Day')
        ax.plot(Y1,label='Normal Weekend')
        ax.plot(Y2,label='Holiday')
        pyplot.legend()
        pyplot.show()




    def checkoutliers(self, lastlink, linkcube):
        print "Check out............linkid=%s" % lastlink
        for day, dayvalues in linkcube.items():
            if day < 74:
                # 3.14之前的数据不检查
                continue
            for hour, hourvalues in dayvalues.items():
                # if day >= 153 and hour == 8:
                if day > 145:
                    continue
                for delta, deltavalues in hourvalues.items():
                    tmp = []
                    for i in range(1,15,1):
                        try:
                            tmp.append(linkcube[day-i][hour][delta]['traveltime'])
                        except:
                            pass
                    if len(tmp) == 0:
                        continue
                    lim1 = np.mean(tmp) + 2*(np.std(tmp))
                    lim2 = np.mean(tmp) - 2*(np.std(tmp))
                    if deltavalues['traveltime'] > lim1 and lim1 > 0:
                        linkcube[day][hour][delta]['traveltime'] = lim1
                    if deltavalues['traveltime'] < lim2 and lim2 > 0:
                        linkcube[day][hour][delta]['traveltime'] = lim2
        return linkcube


    def linkid_onehot(self,x):
        arr = [0]*132
        arr[x]=1
        return arr

    def weekday_onehot(self,x):
        arr = [0]*7
        arr[x]=1
        return arr

    def isholiday_onehot(self,x):
        arr = [0]*3
        arr[x]=1
        return arr

    def print_out(self, lastlinkid, linkcube):
        """输出到文件, linkcube > dayid, hourid, deltaid, values"""
        print "Processing..............linkid=%s" % lastlinkid
        stat_level, stat= self.get_level(linkcube)

        print lastlinkid, stat_level
        for day, dayvalue in linkcube.items():
            if day < 104:
                # 4.13日之前的数据
                continue
            #历史缺失暂时用0填充
            lastmonthdays = set()
            for daydeta in range(30,44,1):
                # [30,31..43] 14 间隔1个月的前2周 当前deltaid
                targetday = day - daydeta
                lastmonthdays.add(targetday)

            lastmonthdays = sorted(lastmonthdays) # 14个dayid

            for hourid, houridvalue in dayvalue.items():
                if hourid not in [7,8,9]:
                    continue
                for deltaid, deltaidvalue in houridvalue.items():
                    # 数据扩充 day > 152 6.1
                    if day > 152 and hourid in [7,9]:   #6月7点和9点的数据无法加工
                        continue
                    lastmonth_values = []
                    # 间隔一个月 跨度2周的  当前deltaid
                    for lastmonthday in lastmonthdays:
                        if lastmonthday not in linkcube or hourid not in linkcube[lastmonthday] or deltaid not in linkcube[lastmonthday][hourid]:
                            lastmonth_values.append(0)
                        else:
                            lastmonth_values.append(linkcube[lastmonthday][hourid][deltaid]['traveltime'])

                    # 抛弃timeseries  只考虑当天的历史数据
                    Today_series = []
                    for delta_id in range(self.secs,-1,-1):
                        realdeltaid = deltaid - delta_id
                        realhourid = hourid - 1
                        if realdeltaid < 0:
                            realdeltaid += 30
                            realhourid -= 1
                        if realhourid not in linkcube[day] or realdeltaid not in linkcube[day][realhourid]:
                            Today_series.append(0)
                        else:
                            Today_series.append(linkcube[day][realhourid][realdeltaid]['traveltime'])

                    # 序列构建

                    sample = [deltaidvalue['traveltime']]  # label
                    sample += [stat]  # 不用于模型训练

                    sample += self.process_today_series(Today_series)   # 当天前一个小时的数据统计特征
                    sample += Today_series

                    sample += lastmonth_values
                    mean_arr, median_arr = self.slide_window(lastmonth_values)
                    sample += mean_arr
                    sample += median_arr

                    # 添加天气信息
                    sample += [self.weather[day][hourid]['SSD'],self.weather[day][hourid]['VIS']]
                    # 周几  是否节假日 Linkid ONE HOT
                    sample += self.linkid_onehot(deltaidvalue['linkid'])
                    sample += self.weekday_onehot(deltaidvalue['weekdayid'])
                    sample += self.isholiday_onehot(deltaidvalue['isholiday'])

                    # 加入组合标识信息，不用于模型训练
                    sample += [deltaidvalue['linkid'], deltaidvalue['dayid'], deltaidvalue['hourid'], deltaidvalue['deltaid']]

                    # self.samples.setdefault(stat_level, {})
                    # self.samples[stat_level].setdefault(hourid, {})
                    # self.samples[stat_level][hourid].setdefault(deltaid, {})
                    # self.samples[stat_level][hourid][deltaid].setdefault(day, [])
                    # self.samples[stat_level][hourid][deltaid][day].append(sample)

                    self.samples.setdefault(lastlinkid, {})
                    self.samples[lastlinkid].setdefault(hourid,{})
                    self.samples[lastlinkid][hourid].setdefault(deltaid,{})
                    self.samples[lastlinkid][hourid][deltaid].setdefault(day,[])
                    self.samples[lastlinkid][hourid][deltaid][day].append(sample)


    def process(self, infile, get_submit=False):
        """linkcube格式:dayid,weekdayid,deltaid,travel_time的dict形式 """
        lastlinkid = ''
        linkcube = {}
        count = 0
        for line in infile:
            if count % 100000 == 0:
                print count
            count += 1
            terms = line.strip().split(';')

            linkid = int(terms[0])

            # dayid weekdayid  >> int
            time_interval = terms[2]
            hourid = self.get_hourid(time_interval) # [0,23]
            if hourid not in [6,7,8,9]:
                continue
            deltaid = self.get_deltaid(time_interval) # [0,29]
            # hourid deltaid >> int

            dateinfo = terms[1]
            dayid, weekdayid = self.get_dayid_weekdayis(dateinfo)
            isholiday = self.get_isholiday(dateinfo)

            travel_time = float(terms[3])

            link_length = float(terms[4])
            link_width = float(terms[5])

            if linkid != lastlinkid:  # 新的Linkid
                if not linkcube:    #第1个linkid
                    linkcube = {}
                    lastlinkid = linkid

                    if dayid not in linkcube:
                        linkcube[dayid] = {}
                    if hourid not in linkcube[dayid]:
                        linkcube[dayid][hourid] = {}

                    linkcube[dayid][hourid][deltaid] = {'weekdayid':weekdayid, 'isholiday':isholiday, 'length':link_length, 'width':link_width,
                                                        'traveltime':travel_time, 'linkid':linkid, 'dayid':dayid,'hourid':hourid, 'deltaid':deltaid}
                    continue

                linkcube = self.checkoutliers(lastlinkid, linkcube)
                self.print_out(lastlinkid, linkcube)
                # self.link_plot(lastlinkid, linkcube, deltaid=0)
                # self.explore_holiday_trend(lastlinkid,linkcube)
                linkcube = {}
                lastlinkid = linkid

            if dayid not in linkcube:
                linkcube[dayid] = {}
            if hourid not in linkcube[dayid]:
                linkcube[dayid][hourid] = {}

            linkcube[dayid][hourid][deltaid] = {'weekdayid':weekdayid, 'isholiday':isholiday,'length':link_length,'width':link_width,
                                                'traveltime':travel_time, 'linkid':linkid, 'dayid':dayid, 'hourid':hourid, 'deltaid':deltaid}

        linkcube = self.checkoutliers(lastlinkid, linkcube)
        self.print_out(lastlinkid, linkcube)
        # self.link_plot(lastlinkid, linkcube, deltaid=0)
        # self.explore_holiday_trend(lastlinkid,linkcube)
        return 0




    def cal_mape(self, result ,thresold):
        """计算mape"""
        mapes = []
        for predict, label in result:
            if label > thresold:
                mape = abs(predict - label) / label
                mapes.append(mape)
        return sum(mapes) / len(mapes)



    def process_relatedfields(self,x):
        """更正dateid, hourid, deltaid显示bug"""
        return int(x)


    def save_to_file(self, keyid, pre, filename):
        """产出提交文件"""
        filepath = os.getcwd()+'\\'+'result'
        pre = np.array(pre)
        dict_value = {
                      # 'linkid':map(self.decode_linkid, keyid[:,0]),
                      'linkid':le.inverse_transform(map(int,keyid[:,0])),
                      'dateid':map(self.process_relatedfields, keyid[:,1]),
                      'hourid':map(self.process_relatedfields, keyid[:,2]),
                      'deltaid':map(self.process_relatedfields, keyid[:,3]),
                      'value':np.exp(pre)-1}
        res = pd.DataFrame(dict_value)
        res.to_csv(filepath+'\\'+'%s.txt'%filename, header=None, index=None, sep='#', columns=['linkid','dateid','hourid','deltaid','value'])
        return 0





    def train(self, get_submit=False):
        training_errors, evaling_errors, baseline_errors = [], [], []

        # samples >> stat_level, hourid, deltaid, day
        for linkid, linkid_vaules in self.samples.items():
            # 分道路建模
            training,evaling,testing = [],[],[]
            for hour, hourvalues in linkid_vaules.items():
                print "linkid : %s. hour :%s" % (linkid, hour)
                for delta, deltavaules in hourvalues.items():
                    for day,dayvalues in deltavaules.items():
                        # 2016-04-13 >> 104    2016-06-30 >> 182  2016-05-31 >> 152

                        # 线上提交：2016-04-13 ~ 2016-05-31数据进行模型的训练，最后7天做模型eval_local

                        if 104<=day<=145:
                            training += dayvalues
                        if 145<day<=152:
                            evaling += dayvalues
                        # 线上提交时,2016-06-01~2016-06-30 早上8:00~9:00的数据进入testing_online
                        if 153<=day<=182 and hour == 8:
                            testing += dayvalues


            print "#######training size is %s. evaling size is %s.########" % (len(training), len(evaling))
            print "#####testing  size: %s.########" % len(testing)


            # 最后4列是组合标识信息  linkid dayid hourid deltaid
            training_x, training_y = np.array(training)[:,1:-4], np.log(np.array(training)[:,0]+1)
            evaling_x, evaling_y = np.array(evaling)[:,1:-4], np.log(np.array(evaling)[:,0]+1)
            baseline_y = np.array(evaling)[:,1]   # baseline_y >> stat均值

            testing_x = np.array(testing)[:,1:-4]

            # print "training_x shape is %s. training_y shape is %s." % (training_x.shape, training_y.shape)
            # print "####################################################"
            # print "evaling_x shape is %s. evaling_y shape is %s." % (evaling_x.shape, evaling_y.shape)

            # predict_train, predict_eval, predict_test = self.xgb_online_process(training_x,training_y,training_online_x,training_online_y,evaling_x,evaling_y,testing_x, 8, 500, 0.8)
            reg, predict_train, predict_eval, predict_test = self.xgb_process(training_x, training_y, evaling_x,evaling_y, testing_x, 8, 500, 0.8)


            # for value in zip(np.exp(predict_eval)-1, np.exp(evaling_y)-1, baseline_y):
            #     print "pre_value, eval_value, baseline: %s\t%s\t%s" % (level, hour, '\t'.join([str(x) for x in value]))


            training_error = self.cal_mape(zip(np.exp(predict_train)-1, np.exp(training_y)-1), 0)
            evaling_error = self.cal_mape(zip(np.exp(predict_eval)-1, np.exp(evaling_y)-1), 0)
            baseline_error = self.cal_mape(zip(baseline_y, np.exp(evaling_y)-1), 0)


            training_errors.append([training_error*len(training_y), len(training_y)])
            evaling_errors.append([evaling_error*len(evaling_y), len(evaling_y)])
            baseline_errors.append([baseline_error*len(evaling_y), len(evaling_y)])


            print "linkid=%s, training_error=%.3f, eavling_error=%.3f, baseline_error=%.3f" % \
                  (linkid, training_error, evaling_error, baseline_error)


            # 输出提交文件
            if get_submit:
                keyinfo = np.array(testing)[:, -4:]
                self.save_to_file(keyinfo, predict_test, filename='pre_linkid_'+str(linkid))

        try:
            print "total errors: training_errors=%.3f, eavling_errors=%.3f, baseline_errors=%.3f" % \
                  (
                      sum([x[0] for x in training_errors]) / sum([x[1] for x in training_errors]),
                      sum([x[0] for x in evaling_errors]) / sum([x[1] for x in evaling_errors]),
                      sum([x[0] for x in baseline_errors]) / sum([x[1] for x in baseline_errors])
                  )
        except:
            print "Can not calculate the total errors."

        return 0


    def xgb_process(self, X, Y, evalX, evalY, testX, depth, iterations, colsample_bytree):
        training_set = xgb.DMatrix(X, label=Y, missing=0)
        evaling_set = xgb.DMatrix(evalX, label=evalY, missing=0)
        testing_set = xgb.DMatrix(testX, missing=0)

        watch_list = [(training_set, 'train'), (evaling_set, 'eval_local')]
        paras = {
            'eta':0.1,
            # 'gamma':0,
            'max_depth':depth,
            # 'min_child_weight':1,
            # 'max_delta_step':0,
            'subsample':0.8,
            'colsample_bytree':colsample_bytree,
            # 'colsample_bylevel':1,
            'lambda':0.1,
            # lambda > l2    alpha > l1
            # 'alpha':0,
            # 'tree_method':'auto',
            # 'sketch_eps':0.03,
            # 'scale_pos_weight':1,
            'objective':'reg:linear',
            'eval_metric':'mae',
            'seed':3,
            'silent':1
        }

        eval_dict = {}
        # tree_booster = xgb.train(params=paras, dtrain=training_set, num_boost_round=iterations, evals=watch_list,
        #                          evals_result=eval_dict, early_stopping_rounds=50)

        tree_booster = xgb.train(
                                params=paras,
                                dtrain=training_set,
                                num_boost_round=iterations,
                                evals=watch_list,
                                evals_result=eval_dict,
                                obj=maelikeobj,
                                feval=eval_log_mape,
                                early_stopping_rounds=50
        )

        predict_train = tree_booster.predict(training_set, output_margin=False, ntree_limit=tree_booster.best_iteration)
        predict_eval = tree_booster.predict(evaling_set, output_margin=False, ntree_limit=tree_booster.best_iteration)
        predict_test = tree_booster.predict(testing_set, output_margin=False, ntree_limit=tree_booster.best_iteration)

        # epochs = len(eval_dict['eval']['mae'])
        # x_axis = range(0, epochs)
        # fig, ax = pyplot.subplots()
        # ax.plot(x_axis, eval_dict['train']['mae'], label='train')
        # ax.plot(x_axis, eval_dict['eval']['mae'], label='eval')
        # ax.legend()
        # pyplot.ylabel('Mae')
        # pyplot.xlabel('epochs')
        # pyplot.title('xgboost mae')
        # pyplot.show()

        return tree_booster, predict_train, predict_eval, predict_test




    def xgb_online_process(self, localX, localY, onlineX, onlineY, evalX, evalY, testX, depth, iterations, colsample_bytree):
        training_local_set = xgb.DMatrix(localX, label=localY,missing=0)
        training_online_set = xgb.DMatrix(onlineX, label=onlineY,missing=0)
        evaling_set = xgb.DMatrix(evalX, label=evalY,missing=0)
        testing_set = xgb.DMatrix(testX,missing=0)

        watch_list = [(training_local_set, 'train_local'), (evaling_set, 'eval_local')]
        paras = {
            'eta':0.1,
            # 'gamma':0,
            'max_depth':depth,
            # 'min_child_weight':1,
            # 'max_delta_step':0,
            'subsample':0.8,
            'colsample_bytree':colsample_bytree,
            # 'colsample_bylevel':1,
            'lambda':0.1,
            # lambda > l2    alpha > l1
            # 'alpha':0,
            # 'tree_method':'auto',
            # 'sketch_eps':0.03,
            # 'scale_pos_weight':1,
            'objective':'reg:linear',
            'eval_metric':'mae',
            'seed':5,
            'silent':1
        }

        eval_dict = {}
        # tree_booster = xgb.train(params=paras, dtrain=training_set, num_boost_round=iterations, evals=watch_list,
        #                          evals_result=eval_dict, early_stopping_rounds=50)

        tree_booster_local = xgb.train(
                                params=paras,
                                dtrain=training_local_set,
                                num_boost_round=iterations,
                                evals=watch_list,
                                evals_result=eval_dict,
                                obj=maelikeobj,
                                feval=eval_log_mape,
                                early_stopping_rounds=50
        )
        best_iteration = tree_booster_local.best_iteration
        tree_booster_online = xgb.train(params=paras, dtrain=training_online_set, num_boost_round=best_iteration, obj=maelikeobj, feval=eval_log_mape)

        predict_local_train = tree_booster_local.predict(training_local_set, output_margin=False, ntree_limit=best_iteration)
        predict_eval = tree_booster_local.predict(evaling_set, output_margin=False, ntree_limit=best_iteration)
        predict_test = tree_booster_online.predict(testing_set, output_margin=False, ntree_limit=best_iteration)

        # epochs = len(eval_dict['eval']['mae'])
        # x_axis = range(0, epochs)
        # fig, ax = pyplot.subplots()
        # ax.plot(x_axis, eval_dict['train']['mae'], label='train')
        # ax.plot(x_axis, eval_dict['eval']['mae'], label='eval')
        # ax.legend()
        # pyplot.ylabel('Mae')
        # pyplot.xlabel('epochs')
        # pyplot.title('xgboost mae')
        # pyplot.show()

        # feature_importance = tree_booster.get_score(importance_type='gain')
        # print "######Feature importance######"
        # print feature_importance
        return predict_local_train, predict_eval, predict_test


    def merge_result(self):
        filedir = os.getcwd()+'\\'+'result'
        filenames = os.listdir(filedir)
        f = open(filedir+'\\'+'result.txt', 'wb')
        for filename in filenames:
            if filename == 'result.txt':
                continue
            filepath = filedir+'\\'+filename
            for line in open(filepath):
                f.writelines(line)
        f.close()
        return 0


    def resort_resultfile(self, filename='result.txt'):
        """按linkid对Merge后的结果文件进行排序,输出到当前路径"""
        filepath = os.getcwd() + '\\' + 'result'
        filename = filepath + '\\' + filename
        data = pd.read_csv(filename, sep='#', header=-1)
        data.columns = ['linkid', 'dayid', 'hourid', 'deltaid', 'value']
        data.sort_values(by=['linkid', 'dayid', 'hourid', 'deltaid'], inplace=True)
        data.to_csv('resort.txt', index=None, header=None)
        return 0


    def combine_sample_resort(self, samplename='tmp_xx.txt', resortname='resort.txt'):
        """最后一步处理,tmp_xx.txt 已经是处理过的数据，按照linkid dayid hourid deltaid排序"""
        f = pd.read_csv(samplename, sep=',', header=-1)
        f.columns = ['linkid', 'dateinfo', 'time_interval', 'dayid', 'hourid', 'deltaid']

        b = pd.read_csv(resortname, sep=',', header=-1)
        b.columns = ['linkid', 'dayid', 'hourid', 'deltaid', 'value']
        print "f shape :%s.   b shape :%s" % (f.shape, b.shape)
        f['value'] = b['value']
        f.to_csv('submit.txt', sep='#', header=None, index=None,
                 columns=['linkid', 'dateinfo', 'time_interval', 'value'])
        return 0


    def plot_stat(self):
        x = self.stats
        # self.stats > list
        fig,ax = pyplot.subplots()
        ax.hist(x,20,cumulative=True,facecolor='pink',alpha=0.8,rwidth=0.8)
        pyplot.show()
        try:
            with open('stats.txt', 'wb') as f:
                for i in x:
                    f.write(str(i)+'\n')
        except:
            print "Can not write stats into TXT."




if __name__ == "__main__":
    weekspan = 4
    dayspan = 5
    secspan = 29

    pdata = ProcessData(weekspan, dayspan, secspan)
    print "######Get Weather Features#####"
    pdata.Weather_Features()
    # infile = 'D:\Pycharm\TC\\new_w_l.txt'
    infile = 'D:\Pycharm\TC\cluster\ClusterRes.txt'
    print "##########Loading##########"
    pdata.process(open(infile))
    print "##########Load done!##########"

    # # pdata.plot_stat()
    # # print "##########Get the Stats!##########"

    print "##########Start training##########:"
    pdata.train(get_submit=True)
    print "##########End training!##########"


    # 线下训练不需要执行
    pdata.merge_result()
    print "##########Get the result.txt!##########"
    pdata.resort_resultfile()
    print "##########Resorted Done!##########"
    pdata.combine_sample_resort()
    print "##########Get Submit.txt!##########"



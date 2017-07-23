#coding:utf-8

import numpy as np
from datetime import datetime
import xgboost as xgb
import cPickle as pickle
import os, os.path
from matplotlib import pyplot
import math
import pandas as pd
from sklearn.preprocessing import LabelEncoder
# 待添加的东西：link_width  link_length  link_class done
# 待改进: encode decode部分优化 done
# 按平均通行时间划分等级 优化  done
# 尝试按speed划分等级 ,评估效果  terrible
# 考虑增加数据


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
    def __init__(self, weekspan, dayspan, hourspan, secspan):
        self.weeks = weekspan
        self.days = dayspan
        self.hours = hourspan
        self.secs = secspan
        self.samples = {}
        # hour 7
        # self.levels = [0,5,11,22,50,200]

        # hour 8
        self.levels = [0,6.4,15.5,32,60,200]

        self.speed_level = [1,5,10,20,100]
        self.stats = []
        self.speed_stats = []
        # [0,5] [5,11] [11,22] [22,50] [50,200]
        # last version [0,10] [10,20] [20,50] [50,100] [100,10000]

    def get_level(self, linkcube):
        """根据道路的历史通过时间划分等级，lincube[dayid][hourid][deltaid]"""
        tmp = []
        for day, dayvalue in linkcube.items():
            # 6月的数据不进入
            if day > 152:
                continue
            for hour,hourvalue in dayvalue.items():
                # 只考虑早上6:00~9:00的平均通行时间
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


    def get_speed_level(self, linkcube):
        """根据道路的平均速度划分等级"""
        tmp = []
        for day, dayvalues in linkcube.items():
            # 6月数据不进入
            if day > 152:
                continue
            for hour, hourvalues in dayvalues.items():
                if hour != 7:
                    continue
                for delta, deltavalues in hourvalues.items():
                    tmp.append(deltavalues['speed'])
        speedstat = sum(tmp) / len(tmp)
        self.speed_stats.append(speedstat)
        print "speed stat.....%.3f" % speedstat
        return
        # for idx in range(len(self.speed_level)-1):
        #     if self.speed_level[idx] <= speedstat < self.speed_level[idx+1]:
        #         return idx, speedstat


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
        """0 > 不是节假日  1 > 是节假日"""
        dayid, weekdayid = self.get_dayid_weekdayis(date_info)
        if date_info in ('2016-04-04','2016-05-02','2016-06-09','2016-06-10'):
            return 1
        if date_info == '2016-06-12':
            return 0
        if weekdayid == 0 or weekdayid == 6:
            return 1
        else:
            return 0


    def slide_window(self, values):
        """滑窗平均值"""
        mean_arr = []
        for i in range(len(values)-1):
            tmp = values[i:i+2]
            mean_arr.append(np.mean(tmp))
        return mean_arr



    def print_out(self, lastlinkid, linkcube):
        """输出到文件, linkcube > dayid, hourid, deltaid, values"""
        print "Processing..............linkid=%s" % lastlinkid
        stat_level, stat= self.get_level(linkcube)

        # self.get_speed_level(linkcube)
        # delta_mean = self.get_delta_mean(linkcube)

        print lastlinkid, stat_level
        for day, dayvalue in linkcube.items():
            #历史缺失暂时用0填充
            # 7/21 update 历史缺失用均值填充
            historydays = set()
            weekdays = set()
            for daydeta in range(self.weeks,0,-1):
                # 前4周的数据, daydeta [4,3,2,1]
                targetday = day - daydeta*7
                weekdays.add(targetday)
                historydays.add(targetday)

            for daydeta in range(self.days,0,-1):
                # 前5天的数据, daydeta [5,4,3,2,1]
                targetday = day - daydeta
                historydays.add(targetday)

            historydays = sorted(historydays) # 周级别和天级别的历史数据,共9个dayid

            for hourid, houridvalue in dayvalue.items():
                for deltaid, deltaidvalue in houridvalue.items():

                    timeseries = []
                    weekdays_value = []   #存放周级别的历史数据
                    for weekday in weekdays:
                        if not weekday in linkcube or not hourid in linkcube[weekday] or deltaid not in linkcube[weekday][hourid]:
                            # weekdays_value.append(delta_mean[hourid][deltaid])
                            weekdays_value.append(0)
                        else:
                            weekdays_value.append(linkcube[weekday][hourid][deltaid]['traveltime'])
                    mean_value = sum(weekdays_value) / len(weekdays_value)
                    # 滑窗均值操作
                    slide_values = self.slide_window(weekdays_value)
                    timeseries.append(mean_value)

                    for historyday in historydays:
                        for delta_id in range(self.secs-1,-1,-1):
                            # self.secs = 15   >> [14,13,...,3,2,1,0]  间隔是2mins，之前0.5个小时的数据
                            realdeltaid = deltaid - delta_id
                            realhourid = hourid
                            # 1小时按2min划分，delta >> [0,29]
                            realday = historyday

                            if realdeltaid < 0:
                                realdeltaid += 30
                                realhourid -= 1
                            if realhourid < 0:   # hourid [0,23]
                                realhourid += 24
                                realday -= 1

                            if realday not in linkcube or realhourid not in linkcube[realday] or realdeltaid not in linkcube[realday][realhourid]:
                                # timeseries.append(delta_mean[realhourid][realdeltaid])
                                timeseries.append(0)
                            else:
                                timeseries.append(linkcube[realday][realhourid][realdeltaid]['traveltime'])

                    hour_values = []  # 存放小时级别的历史数据
                    for i in range(self.hours,0,-1):
                        # self.hours = 1, 前一小时数据
                        realhourid = hourid - i
                        realday = day
                        if realhourid < 0:
                            realhourid += 24
                            realday -= 1
                        if realday not in linkcube or realhourid not in linkcube[realday] or deltaid not in linkcube[realday][realhourid]:
                            # hour_values.append(delta_mean[realhourid][deltaid])
                            hour_values.append(0)
                        else:
                            hour_values.append(linkcube[realday][realhourid][deltaid]['traveltime'])


                    mins_values = [] # 存放当前的前30分钟的数据
                    for i in range(self.secs,0,-1):
                        # self.secs = 15   [15,14,...,1]
                        realdeltaid = deltaid - i
                        realhourid = hourid
                        realday = day
                        if realdeltaid < 0:
                            realdeltaid += 30
                            realhourid -= 1
                        if realhourid < 0:
                            realhourid += 24
                            realday -= 1
                        if realday not in linkcube or realhourid not in linkcube[realday] or realdeltaid not in linkcube[realday][realhourid]:
                            # mins_values.append(delta_mean[realhourid][realdeltaid])
                            mins_values.append(0)
                        else:
                            mins_values.append(linkcube[realday][realhourid][realdeltaid]['traveltime'])


                    # 序列构建
                    # timeseries hour_values mins_values:{前4周的均值,9个dayid 对应的前0.5小时数据,当前1个间隔一小时的数据,当前半小时的数据} 共 1+9*15+1+15=152个数据
                    # sample:{真实值,timeseries,weekdayid,isholiday} 155个数据+4个标识信息 = 159 + 3个slide values = 162 + length width = 164
                    sample = [deltaidvalue['traveltime']]
                    sample += timeseries
                    sample += slide_values
                    sample += hour_values
                    sample += mins_values
                    # 周几的标记， 节假日标记
                    sample += [deltaidvalue['weekdayid'], deltaidvalue['isholiday']]
                    # 加入link length width
                    sample += [deltaidvalue['length'], deltaidvalue['width']]
                    # 加入平均速度  和 当前速度
                    sample += [float(stat / deltaidvalue['length'])]

                    # sample += [deltaidvalue['speed']]

                    # 加入组合标识信息
                    sample += [deltaidvalue['linkid'], deltaidvalue['dayid'], deltaidvalue['hourid'], deltaidvalue['deltaid']]


                    self.samples.setdefault(stat_level, {})
                    self.samples[stat_level].setdefault(hourid, {})
                    self.samples[stat_level][hourid].setdefault(deltaid, {})
                    self.samples[stat_level][hourid][deltaid].setdefault(day, [])
                    self.samples[stat_level][hourid][deltaid][day].append(sample)


    def encode_linkid(self, x):
        """编码 x > str"""
        data = pd.read_csv('gy_contest_link_info.txt', sep=';')
        le = LabelEncoder()
        le.fit(data['link_ID'])
        return le.transform([x])[0]



    def decode_linkid(self, x):
        """反编码,x > int"""
        data = pd.read_csv('gy_contest_link_info.txt', sep=';')
        le = LabelEncoder()
        le.fit(data['link_ID'])
        return le.inverse_transform([int(x)])[0]




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
            if get_submit:
                # 直接对linkid进行编码, linkid > int
                # linkid = self.encode_linkid(terms[0])
                linkid = le.transform([terms[0]])[0]
            if not get_submit:
                linkid = int(terms[0])

            dateinfo = terms[1]
            dayid, weekdayid = self.get_dayid_weekdayis(dateinfo)
            isholiday = self.get_isholiday(dateinfo)
            # dayid weekdayid  >> int
            time_interval = terms[2]
            # 一条记录的唯一标识  linkid  dayid hourid deltaid 组合标识

            hourid = self.get_hourid(time_interval) # [0,23]
            deltaid = self.get_deltaid(time_interval) # [0,29]
            # hourid deltaid >> int
            travel_time = float(terms[3])

            link_length = float(terms[4])
            link_width = float(terms[5])

            speed = travel_time / link_length

            if linkid != lastlinkid:  # 新的Linkid
                if not linkcube:    #第1个linkid
                    linkcube = {}
                    lastlinkid = linkid

                    if dayid not in linkcube:
                        linkcube[dayid] = {}
                    if hourid not in linkcube[dayid]:
                        linkcube[dayid][hourid] = {}

                    linkcube[dayid][hourid][deltaid] = {'weekdayid':weekdayid, 'isholiday':isholiday, 'length':link_length, 'width':link_width,
                                                        'traveltime':travel_time, 'linkid':linkid, 'dayid':dayid,'hourid':hourid, 'deltaid':deltaid, 'speed':speed}
                    continue
                self.print_out(lastlinkid, linkcube)
                linkcube = {}
                lastlinkid = linkid

            if dayid not in linkcube:
                linkcube[dayid] = {}
            if hourid not in linkcube[dayid]:
                linkcube[dayid][hourid] = {}

            linkcube[dayid][hourid][deltaid] = {'weekdayid':weekdayid, 'isholiday':isholiday,'length':link_length,'width':link_width,
                                                'traveltime':travel_time, 'linkid':linkid, 'dayid':dayid, 'hourid':hourid, 'deltaid':deltaid, 'speed':speed}

        self.print_out(lastlinkid, linkcube)
        return 0


    def maybe_pickle(self, infile):
        pickle_file = '%s.pickle' % (infile)
        self.pickle_file = pickle_file
        if os.path.exists(self.pickle_file):
            print "%s already exists.....skipping pickle" % pickle_file
        else:
            print "Pickling %s" % pickle_file
            self.process(open(infile))
            try:
                with open(pickle_file,'wb') as f:
                    pickle.dump(self.samples, f, pickle.HIGHEST_PROTOCOL)
            except Exception as e:
                print "Unable to save data to %s" % pickle_file


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


    def train(self):
        # with open(self.pickle_file, 'rb') as f:
        #     print "load data from %s."% self.pickle_file
        #     self.samples = pickle.load(f)

        training_errors, evaling_errors, baseline_errors = [], [], []
        testing_errors = []
        # samples >> stat_level, hourid, deltaid, day
        for level, level_vaules in self.samples.items():
            for hour, hourvalues in level_vaules.items():
                # 分小时建模，不需要针对诸如22：00等时间建模
                print "level : %s. hour :%s" % (level, hour)
                # if hour != 7:
                #     continue
                if hour != 8:
                    continue

                training = []
                evaling = []
                testing = []
                for delta, deltavaules in hourvalues.items():
                    for day,dayvalues in deltavaules.items():
                        # 2016-03-01 >> 61    2016-06-30 >> 182  2016-05-31 >> 152

                        # 线上提交：2016-03-01 ~ 2016-05-31数据进行模型的训练，最后5天做模型eval_local

                        if 61<=day<=147:
                            training += dayvalues
                        if 147<day<=152:
                            evaling += dayvalues

                        # 线上提交时,2016-06-01~2016-06-30 早上8:00~9:00的数据进入testing_online

                        if 153<=day<=182:
                            testing += dayvalues


                print "training size is %s. evaling size is %s." % (len(training), len(evaling))

                print "#####testing  size: %s.########" % len(testing)
                # 分小时建模


                if len(testing) == 0:
                    continue


                # x > 142   y > 1
                # 加入组合标识信息后  x > 146
                # 最后4列是组合标识信息  linkid dayid hourid deltaid
                training_x, training_y = np.array(training)[:,1:-4], np.log(np.array(training)[:,0]+1)
                evaling_x, evaling_y = np.array(evaling)[:,1:-4], np.log(np.array(evaling)[:,0]+1)
                baseline_y = np.array(evaling)[:,1]   # baseline_y >> 前4周的均值

                testing_x = np.array(testing)[:,1:-4]
                if hour == 7:
                    testing_y = np.log(np.array(testing)[:,0]+1)

                # 产生提交数据才需要保存keyinfo
                if hour == 8:
                    keyinfo = np.array(testing)[:,-4:]

                print "training_x shape is %s. training_y shape is %s." % (training_x.shape, training_y.shape)
                print "####################################################"
                print "evaling_x shape is %s. evaling_y shape is %s." % (evaling_x.shape, evaling_y.shape)

                # reg, predict_train, predict_test = self.xgb_local_process(training_x, training_y, testing_local_x,testing_local_y, 4, 500, 0.8)

            	reg, predict_train, predict_eval, predict_test = self.xgb_online_process(training_x, training_y, evaling_x, evaling_y, testing_x, 4, 500, 0.8)
            

                for value in zip(np.exp(predict_eval)-1, np.exp(evaling_y)-1, baseline_y):
                    print "pre_value, eval_value, baseline: %s\t%s\t%s" % (level, hour, '\t'.join([str(x) for x in value]))


                if hour == 7:
                    for value in zip(np.exp(predict_test)-1, np.exp(testing_y)-1):
                        print "Model Get: %s\t%s\t%s" % (level, hour, '\t'.join([str(x) for x in value]))

                # 计算误差时已经完成了log 与 exp的操作
                training_error = self.cal_mape(zip(np.exp(predict_train)-1, np.exp(training_y)-1), 0)
                evaling_error = self.cal_mape(zip(np.exp(predict_eval)-1, np.exp(evaling_y)-1), 0)
                baseline_error = self.cal_mape(zip(baseline_y, np.exp(evaling_y)-1), 0)

                if hour == 7:
                    testing_error = self.cal_mape(zip(np.exp(predict_test)-1, np.exp(testing_y)-1),0)
                    testing_errors.append([testing_error*len(testing_y), len(testing_y)])

                training_errors.append([training_error*len(training_y), len(training_y)])
                evaling_errors.append([evaling_error*len(evaling_y), len(evaling_y)])
                baseline_errors.append([baseline_error*len(evaling_y), len(evaling_y)])

                if hour == 8:
                    print "level=%s, hour=%s, training_error=%.3f, eavling_error=%.3f, baseline_error=%.3f" % \
	                      (level, hour, training_error, evaling_error, baseline_error)
                if hour == 7:
                    print "levle=%s, hour=%s, training_error=%.3f, evaling_error=%.3f, baseline_error=%.3f, testing_error=%.3f" % \
              			(level, hour, training_error, evaling_error, baseline_error, testing_error)

                # 输出提交文件
                if hour == 8:
                    self.save_to_file(keyinfo, predict_test, filename='pre_level_'+str(level)+'hour_'+str(hour))

        if testing_errors:
            print "total error: training_error=%.3f, evaling_error=%.3f, testing_error=%.3f" % \
    			  (
 					  sum([x[0] for x in training_errors]) / sum([x[1] for x in training_errors]),
                      sum([x[0] for x in evaling_errors]) / sum([x[1] for x in evaling_errors]),
                      sum([x[0] for x in testing_errors]) / sum([x[1] for x in testing_errors])
    			  	)

        try:
            print "total error: training_error=%.3f, eavling_error=%.3f, baseline_error=%.3f" % \
                  (
                      sum([x[0] for x in training_errors]) / sum([x[1] for x in training_errors]),
                      sum([x[0] for x in evaling_errors]) / sum([x[1] for x in evaling_errors]),
                      sum([x[0] for x in baseline_errors]) / sum([x[1] for x in baseline_errors])
                  )
        except:
            print "Can not calculate the total error."

        try:

            print "level 1+ error: training_error=%.3f, evaling_error=%.3f, baseline_error=%.3f" % \
              (
                  sum([x[0] for x in training_errors[1:]]) / sum([x[1] for x in training_errors[1:]]),
                  sum([x[0] for x in evaling_errors[1:]]) / sum([x[1] for x in evaling_errors[1:]]),
                  sum([x[0] for x in baseline_errors[1:]]) / sum([x[1] for x in baseline_errors[1:]])
              )
        except:
            print "Can not calculate the level 1+ error."

        try:
            print "level 2+ error: training_error:%.3f, evaling_error=%.3f, baseline_error=%.3f" % \
                  (
                      sum([x[0] for x in training_errors[2:]]) / sum([x[1] for x in training_errors[2:]]),
                      sum([x[0] for x in evaling_errors[2:]]) / sum([x[1] for x in evaling_errors[2:]]),
                      sum([x[0] for x in baseline_errors[2:]]) / sum([x[1] for x in baseline_errors[2:]])
                  )
        except:
            print "Can not calculate the level 2+ error."
        return 0


    def xgb_online_process(self, X, Y, evalX, evalY, testX, depth, iterations, colsample_bytree):
        training_set = xgb.DMatrix(X, label=Y)
        evaling_set = xgb.DMatrix(evalX, label=evalY)
        testing_set = xgb.DMatrix(testX)

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
            'seed':5,
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


    def get_speed_stat_file(self):
        x = self.speed_stats
        try:
            with open('speed_stats.txt', 'wb') as f:
                for i in x:
                    f.write(str(i)+'\n')
        except:
            print "Can not write speed stats into TXT."




if __name__ == "__main__":
    weekspan = 4
    dayspan = 5
    hourspan = 3
    secspan = 15

    pdata = ProcessData(weekspan, dayspan, hourspan, secspan)
    # 1M rows
    # infile = 'D:\Pycharm\TC\mini.txt'

    infile = 'D:\Pycharm\TC\\new_w_l.txt'
    print "##########Loading##########"
    pdata.process(open(infile), get_submit=True)
    # pdata.maybe_pickle(infile)
    print "##########Load done!##########"

    # pdata.plot_stat()
    # print "##########Get the Stats!##########"

    print "##########Start training##########:"
    pdata.train()
    print "##########End training!##########"


    # 线下训练不需要执行
    pdata.merge_result()
    print "##########Get the result.txt!##########"
    pdata.resort_resultfile()
    print "##########Resorted Done!##########"
    pdata.combine_sample_resort()
    print "##########Get Submit.txt!##########"




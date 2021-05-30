import pdb
import numpy as np
import pandas as pd

## membership status
years      = ['2014', '2015', '2016', '2017', '2018']
userStatus = ['bronze', 'silver', 'gold', 'inactive']

user_years = np.random.choice(years, 1000, p = [0.1, 0.1, 0.15, 0.3, 0.35])
user_stats = np.random.choice(userStatus, 1000, p = [0.5, 0.3, 0.1, 0.1])

year_joined = pd.DataFrame({'year_joined': user_years,
                            'userStats': user_stats})

num_emails_sent_weekly = 3


def never_open_rate(period_rng):
    return []

# 조사기간 중 뉴스레터 열람횟수는? (간격 week)
def constant_open_rate(period_rng):
    n, p = num_emails_sent_weekly, np.random.uniform(0, 1)
    num_opened = np.random.binomial(n, p, len(period_rng))
    return num_opened


def open_rate_with_factor_change(period_rng, fac):
    
    if len(period_rng) < 1:
        return []
    
    # 메일을 열지 않는 주 추가(10% 확률)
    times = np.random.randint(0, len(period_rng), int(0.1 * len(period_rng)))
    try:
        n, p = num_emails_sent_weekly, np.random.uniform(0, 1)
        num_opened = np.zeros(len(period_rng))
        for pd in range(0, len(period_rng), 2):
            num_opened[pd:(pd+2)] = np.random.binomial(n, p, 2)
            # 뉴스레터를 신청했으나 2주마다 열람확률 감소
            p = max(min(1, p * fac), 0)
    except:
        num_opened[pd] = np.random.binomial(n, p, 1)
    for t in times:
        num_opened[t] = 0
    return num_opened

def increasing_open_rate(period_rng):
    return open_rate_with_factor_change(period_rng, np.random.uniform(1.01, 1.03))

def decreasing_open_rate(period_rng):
    return open_rate_with_factor_change(period_rng, np.random.uniform(0.5, 0.99))



# 기부 함수 용
def random_weekly_time_delta():
    days_of_week = [d for d in range(7)]
    
    # 일반적으로 기부를 새벽에 하지는 않을 것 아냐 9~10시지
    hours_of_day = [h for h in range(11, 23)]
    minute_of_hour = [m for m in range(60)]
    second_of_minute = [s for s in range(60)]
    return pd.Timedelta(str(np.random.choice(days_of_week))     + "days")    + \
           pd.Timedelta(str(np.random.choice(hours_of_day))    + "hours")   + \
           pd.Timedelta(str(np.random.choice(minute_of_hour))   + "minutes") + \
           pd.Timedelta(str(np.random.choice(second_of_minute)) + "seconds")



def produce_donations(period_rng, user_behavior, num_emails, use_id, user_join_year):
    donation_amounts = np.array([0, 25, 50, 75, 100, 250, 500, 1000, 1500, 2000])
    user_has = np.random.choice(donation_amounts)
    
    # num_emails란?
    # 이메일 안 열어본 놈, 50주동안 변함없는 확률로 열어보는 놈, 초반에는 잘 열다가 나중에 안 여는 놈, 갈 수록 혼모노가 되어가는 놈 중 하나를 랜덤으로 택해서 email을 open한 횟수를 다 더한 값이다 
    # user_gives란?
    # 고객의 이메일 열람 횟수 합/회사에서 이메일 보낸 수 * 기부금액
    user_gives = num_emails / (num_emails_sent_weekly * len(period_rng)) * user_has
    
    # np.where쓰면 조건에 맞는 어레이 인덱스를 알려준다
    # 그 인덱스의 첫 번째 행, 마지막 인덱스 값을 선택한다는 뜻
    # 즉, 특정 유저의 기부 금액 중 가장 큰 금액을 뽑아낸다
    user_gives_idx = np.where(user_gives >= donation_amounts)[0][-1]
    user_gives_idx = max(min(user_gives_idx, len(donation_amounts) - 2), 1)
    
    # 기부는 한 명당 평균 1년에 2번정도는 하더라
    num_times_gave = np.random.poisson(2) * (2018 - user_join_year)
    
    times =  np.random.randint(0, len(period_rng), num_times_gave)
    
    # 빈 깍을 만들어둬야 for문 안에서 사라지지 않음
    donations = pd.DataFrame({'user': [], 'amount': [], 'timestamp': []})
    for n in range(num_times_gave):
        
        donations = donations.append(pd.DataFrame({'user': [use_id], 'amount': [donation_amounts[user_gives_idx + np.random.binomial(1, .3)]], 'timestamp': [str(period_rng[times[n]].start_time + random_weekly_time_delta())]}))
    
    if donations.shape[0] > 0:
        donations = donations[donations.amount != 0]
    return donations



## run it!!!
behaviors = [never_open_rate, constant_open_rate, increasing_open_rate, decreasing_open_rate]
user_behaviors = np.random.choice(behaviors, 1000, [0.2, 0.5, 0.1, 0.2])

rng = pd.period_range('2015-02-14', '2018-06-01', freq = 'W')
emails = pd.DataFrame({'user': [], 'week': [], 'emailsOpened':[]})
donations = pd.DataFrame({'user': [], 'amount': [], 'timestamp':[]})


for idx in range(year_joined.shape[0]):
    
    # user가 가입한 일자를 타임스탬프로 변형
    join_date = pd.Timestamp(year_joined.iloc[idx].year_joined) + pd.Timedelta(str(np.random.randint(0, 365)) + ' days')
    join_date = min(join_date, pd.Timestamp('2018-06-01'))
    
    # 가입한 이후에 기부함
    user_rng = rng[rng.start_time > join_date]
    
    if len(user_rng) < 1:
        continue
    
    # user_behavior는 function object로 이루어진 array다
    info =  user_behaviors[idx](user_rng)
    
    if len(info) == len(user_rng):
        emails = emails.append(pd.DataFrame({
            'user': [idx] * len(info), 
            # datetime index가 2016-06/2016-07 이런식으로
            # 있을 때, 앞부분만 인덱싱 .start_time
            'week': [str(r.start_time) for r in user_rng],
            'emailsOpened': info}))
    donations = donations.append(produce_donations(user_rng, user_behaviors[idx], sum(info), idx, join_date.year))

emails = emails[emails.emailsOpened != 0]
year_joined.index.name = 'user'


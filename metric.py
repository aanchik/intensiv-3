import pandas as pd

def decision_prices(test):  #функция принимает на вход test с данными по арматуре
    test = test.set_index('dt')  #dt становится индексом
    tender_price = test['Цена на арматуру']
    decision = test['Объем']  #колонка с количеством недель закупки
    start_date = test.index.min() #определение границ временного диапазона
    end_date = test.index.max()
    
    _results = []  #список для хранения цен, по которым закупалась арматура
    _active_weeks = 0
    for report_date in pd.date_range(start_date, end_date, freq='W-MON'):  #проходим по всем понедельникам в диапазоне start_date — end_date.
        if _active_weeks == 0:  # Пришла пора нового тендера
            _fixed_price = tender_price.loc[report_date]  #запоминает цену на арматуру
            _active_weeks = int(decision.loc[report_date])
        _results.append(_fixed_price)
        _active_weeks += -1
    cost = sum(_results)  #суммирует все зафиксированные цены
    return cost #dозвращаем итоговые затраты на арматуру
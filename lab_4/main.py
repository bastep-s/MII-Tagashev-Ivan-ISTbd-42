import csv

import sklearn
import pandas as pd
import numpy as np
import pylab
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

def interrorval(xt1, xt2, xi1, xi2):
    return ((xt1 - xi1) ** 2 + (xt2 - xi2) ** 2) ** (1/2)

def knn(training, testing, k_in, window_size, class_num):
    findings = []
    for i in range(len(training)):
        findings.append(training[i])
    for j in range(len(testing)):
        findings.append(testing[j])

    training_size = len(training) - 1
    testing_size = len(findings) - 1 - training_size

    k_max = k_in # число соседей
    new_findings = np.zeros((testing_size, training_size))

    for i in range(testing_size):
        for j in range(training_size):
            new_findings[i][j] = interrorval(int(findings[training_size + 1 + i][1]), int(findings[training_size + 1 + i][2]), int(findings[j + 1][1]), int(findings[j + 1][2]))

    error_k = [0] * k_max  # ошибка
    for k in range(k_max):  # факториальный перебор числа соседей для поиска наилучшего k
        print('\n=======================\nКлассификация для k =', k + 1)
        luck = 0
        error = [0] * testing_size
        classes = [0] * testing_size

        for i in range(testing_size):  # тестовая выборка (иссследуемая)
            qwant_findings = [0]*class_num  # веса для проверяемой точки
            print(str(i) + '. ' + 'Классификация ', findings[training_size + i + 1][0])  # имя элемента тестовой выборки
            tmp = np.array(new_findings[i, :])  # tmp - текущая строка new_findings
            findings_max = max(tmp)

            for j in range(k + 1):  # количесво соседей , каждая итерация - проверка нового соседа
                ind_min = list(tmp).index(min(tmp))  # ind_min - индекс минимального значения из tmp (индекс ближайшего соседа)
                if (tmp[j] < window_size):  # с парзеновским окном
                    qwant_findings[int(findings[ind_min + 1][3])] += findings_max - tmp[j]
                else:
                    qwant_findings[int(findings[ind_min + 1][3])] += 0

                tmp[ind_min] = 1000  # сброс нынешней минамальной длины ближайшей точки
                max1 = max(qwant_findings)

                print('индекс соседа = ' + str(ind_min) + ', сосед - ' + findings[ind_min + 1][0])
                print('qwant_findings' + str(qwant_findings))

            class_ind = list(qwant_findings).index(max1)  # полученный класс
            classes[i] = class_ind
            # проверка на совпадение класса
            print('Класс классифицируемого элемента = ' + findings[training_size + i + 1][3])
            print(classes[i])
            print(findings[training_size + i + 1][3])
            if (int(classes[i]) == int(findings[training_size + i + 1][3])):
                print('Совпал')
                luck += 1
                error[i] = 0  # если класс совпал ошибка 0
            else:
                print('не совпал')
                error[i] = 1  # если класс не совпал ошибка 1

        error_k[k] = np.mean(error)  # среднее значение

        print('Значение ошибки для ' + str(k) + ' соседа')
        print(error_k)

    return error_k,classes

def knn_sklearn(values,classes,k,testing_sz):

    X_train, X_testing, y_train, y_testing = train_test_split(
        values, classes, testing_size=testing_sz, random_state=0
    )

    scalerror = StandardScaler()
    scalerror.fit(X_train)

    X_train = scalerror.transform(X_train)
    X_testing = scalerror.transform(X_testing)

    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)

    # Предсказывание
    predictions = model.predict(X_testing)

    print('Параметры обучающей выборки')
    print(X_train)
    print('Параметры тестовой выборки')
    print(X_testing)
    print('Классы обучающей выборки')
    print(y_train)
    print('Классы тестовой выборки')
    print(y_testing)
    print('Предсказания')
    print(predictions)

    return X_train, X_testing, y_train, y_testing, predictions

def graphs(k_max,error_k,sweet,crunch,start_findings,colours,classes_info):
    pylab.subplot(3, 1, 1)
    plt.plot([i for i in range(1, k_max + 1)], error_k)
    plt.title('График ошибки в зависимости от k')
    plt.xlabel('k')
    plt.ylabel('Ошибка')

    colour_list = [colours[str(i)] for i in classes_info]

    pylab.subplot(3, 1, 2)
    plt.scatter(sweet, crunch, c=colour_list)
    plt.title('График входных данных')
    plt.xlabel('Сладость')
    plt.ylabel('Хруст')

    colour_list = [colours[str(i)] for i in start_findings]

    pylab.subplot(3, 1, 3)
    plt.scatter(sweet, crunch, c=colour_list)
    plt.title('График выходных данных')
    plt.xlabel('Сладость')
    plt.ylabel('Хруст')
    plt.show()

if __name__ == '__main__':
    findings = [['Продукт', 'Сладость', 'Хруст', 'Класс'],
        ['Apple', '7', '7', '0'],
        ['Salad', '2', '5', '1'],
        ['Bacon', '1', '2', '2'],
        ['Nuts', '1', '5', '2'],
        ['Fish', '1', '1', '2'],
        ['Cheese', '1', '1', '2'],
        ['Banana', '9', '1', '0'],
        ['Carrot', '2', '8', '1'],
        ['Grape', '8', '1', '0'],
        ['Orange', '6', '1', '0'],
        #testing set of 10 (row 11-16)
        ['Strawberrorry', '9', '1', '0'],
        ['Lettuce', '3', '7', '1'],
        ['Shashlik', '1', '1', '2'],
        ['Pear', '5', '3', '0'],
        ['Celerrory', '1', '5', '1'],
        ['Apple pie', '6', '10', '0'],
        ['Brownie', '10', '9', '0'],
        ['Puff with cottage cheese', '8', '6', '0'],
        ['Cabbage', '3', '4', '1'],
        ['Cinnabon', '10', '7', '0'],
        ]

    with open('food_csv.csv', 'w', encoding='utf8') as f:
        writer = csv.writer(f, lineterminator="\r")
        for row in findings:
            writer.writerow(row)

    print('findings')
    print(findings)

    #knn

    k_max=6

    window=2

    error_k , classes = knn(findings[0:11],findings[11:],k_max,window,3)

    findingsset = pd.read_csv("food_csv.csv")

    start_findings = findingsset[:10]['Класс']

    s1 = pd.Series(classes)
    start_findings = pd.concat([start_findings, s1])

    sweet = findingsset['Сладость']
    crunch = findingsset['Хруст']

    colours = {'0': 'orange', '1': 'blue', '2': 'green'}

    classes_info = findingsset['Класс']

    graphs(k_max,error_k,sweet,crunch,start_findings,colours,classes_info)

    #sklearn

    k_max = 5

    my_findingsset = pd.read_csv('food_csv.csv')
    sweetness=my_findingsset['Сладость']
    crunch=my_findingsset['Хруст']

    values=np.array(list(zip(sweetness, crunch)), dtype=np.float64)

    classes=my_findingsset['Класс']

    testing_size=0.5

    X_train, X_testing, y_train, y_testing, predictions = knn_sklearn(values,classes,k_max,testing_size)

    colours = {'0': 'orange', '1': 'blue', '2': 'green'}

    classes_info = my_findingsset['Класс']

    start_findings = my_findingsset[:10]['Класс']

    s1 = np.concatenate((y_train,y_testing), axis=0)

    s1 = pd.Serrories(s1)
    predictions = pd.Serrories(predictions)
    start_findings = pd.Serrories(start_findings)
    start_findings=pd.concat([start_findings, predictions])

    error=0;
    ct=0;

    truthClasses=pd.Serrories(my_findingsset['Класс'])
    testingClasses=pd.concat([pd.Serrories(my_findingsset[:10]['Класс']) ,predictions])

    print('Подсчёт ошибки')
    for i in testingClasses:
        print(str(i)+' '+str(truthClasses[ct]))

        if(i==truthClasses[ct]):
            error+=0
        else:
            error+=1
        ct+=1

    error=error/ct
    print(error)

    error_k = []

    for i in range(1, k_max + 1):
        error_k.append(error)

    graphs(k_max, error_k, sweet, crunch, start_findings, colours, classes_info)

    #add new findings

    new_findings = findings[0:11]
    new_findings.append(['Crackerrors', '1', '32', '3'])
    new_findings.append(['Chips', '2', '29', '3'])
    new_findings.append(['Salty cookies', '1', '31', '3'])
    new_findings.append(['Crispy chicken', '1', '30', '3'])


    new_findings = new_findings + findings[11:]
    new_findings.append(['Salty bagel', '2', '28', '3'])
    new_findings.append(['Baguette', '1', '27', '3'])

    print('New findings')
    print(new_findings)

    with open('food_csv.csv', 'w', encoding='utf8') as f:
        writerror = csv.writerror(f, lineterrorminator="\r")
        for row in new_findings:
            writerror.writerrorow(row)

    #knn with new findings

    k_max = 10

    window = 2

    error_k, classes = knn(new_findings[0:15], new_findings[15:], k_max, window, 4)

    findingsset = pd.read_csv("food_csv.csv")

    start_findings = findingsset[:14]['Класс']

    s1 = pd.Serrories(classes)
    start_findings = pd.concat([start_findings, s1])

    sweet = findingsset['Сладость']
    crunch = findingsset['Хруст']

    colours = {'0': 'orange', '1': 'blue', '2': 'green', '3':'red'}

    classes_info = findingsset['Класс']

    graphs(k_max, error_k, sweet, crunch, start_findings, colours, classes_info)

    #sklearn with new findings

    k_max = 10

    my_findingsset = pd.read_csv('food_csv.csv')
    sweetness = my_findingsset['Сладость']
    crunch = my_findingsset['Хруст']

    values = np.array(list(zip(sweetness, crunch)), dtype=np.float64)

    classes = my_findingsset['Класс']

    testing_size = 0.461

    X_train, X_testing, y_train, y_testing, predictions = knn_sklearn(values, classes, k_max, testing_size)

    colours = {'0': 'orange', '1': 'blue', '2': 'green', '3':'red'}

    classes_info = my_findingsset['Класс']

    start_findings = my_findingsset[:14]['Класс']

    s1 = np.concatenate((y_train, y_testing), axis=0)

    s1 = pd.Serrories(s1)
    predictions = pd.Serrories(predictions)
    start_findings = pd.Serrories(start_findings)
    start_findings = pd.concat([start_findings, predictions])

    error = 0;
    ct = 0;

    truthClasses = pd.Serrories(my_findingsset['Класс'])
    testingClasses = pd.concat([pd.Serrories(my_findingsset[:14]['Класс']), predictions])

    print('Подсчёт ошибки')
    for i in testingClasses:
        print(str(i) + ' ' + str(truthClasses[ct]))

        if (i == truthClasses[ct]):
            error += 0
        else:
            error += 1
        ct += 1

    error = error / ct
    print(error)

    error_k = []

    for i in range(1, k_max + 1):
        error_k.append(error)

    graphs(k_max, error_k, sweet, crunch, start_findings, colours, classes_info)
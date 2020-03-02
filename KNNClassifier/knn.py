from csv import reader
from math import sqrt

train_file = 'train.csv'
test_file = 'test.csv'

def knn(file, k, distance_method):
    #load the csv
    dataset = read_file(file)

    #predict
    predictions = []
    classes = []
    for row in dataset:
        output = predict(dataset, row, k, distance_method)
        classes.append(row[0])
        predictions.append(output)

    #calculate accuracy
    accuracy = calculate_accuracy(classes, predictions)
    #reset classes to empty list
    classes = []
    print("K : ", k)
    print ("Distance_method :", distance_method)
    print('Accuracy : ', accuracy, '%')
    print("----------------------------------------------")

def read_file(file):
    dataset = []
    with open(file, 'r') as csv:
        csv_reader = reader(csv, delimiter=',', quotechar='|')
        index = 0
        for row in csv_reader:
            dataset.append(row)
            index += 1
            if index == 100:
                return dataset
    return dataset 

def euclidean_distance(row1, row2):
    distance = 0.0
    for i in range(1,len(row1)-1):
        distance += (int(row1[i]) - int(row2[i]))**2
    return sqrt(distance)

def manhattan_distance(row1, row2):
    distance = 0.0
    for i in range(1,len(row1)-1):
        distance += (abs(int(row1[i])) - abs(int(row2[i])))
    return distance


def get_nearest_neighbors(dataset, row, k, distance_method):
    #Compute the distance between the row in params and the rest of the dataset
    distances = []
    for curentRow in dataset:
        if distance_method == 'euclidean':
            distance = euclidean_distance(row, curentRow)
        elif distance_method == 'manhattan':
            distance = manhattan_distance(row, curentRow)
        distances.append((distance, curentRow))
    
    #Sort the list
    distances.sort(key=lambda item: item[0])

    #Take the first k neighbors
    neighbors = []
    for i in range(0,k):
        neighbors.append(distances[i])
    
    return neighbors

def predict(dataset, row, k, distance_method):
    neighbors = get_nearest_neighbors(dataset, row, k, distance_method)
    #Get list of classes from neighbors
    classes = [test[1][0] for test in neighbors]
    print(classes)
    #Get the class that appears the most
    prediction = max(set(classes), key=classes.count)
    return prediction

def calculate_accuracy(correct_classes, predicted_classes):
	correct = 0
	for i in range(0, len(correct_classes)-1):
		if correct_classes[i] == predicted_classes[i]:
			correct += 1
	return correct / float(len(correct_classes)) * 100.0

def run_samples():
    #run with euclidean distance
    knn(test_file, 1, 'euclidean')
    knn(test_file, 3, 'euclidean')
    knn(test_file, 5, 'euclidean')
    knn(test_file, 10, 'euclidean')
    knn(test_file, 15, 'euclidean')

    #run with manhattan distance
    knn(test_file, 1, 'manhattan')
    knn(test_file, 3, 'manhattan')
    knn(test_file, 5, 'manhattan')
    knn(test_file, 10, 'manhattan')
    knn(test_file, 15, 'manhattan')


run_samples()
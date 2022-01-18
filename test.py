import matplotlib.pyplot as plt

# classifier = ['Begging','Boosting']
# accuracy = [85.7,87.5]
  
# plt.plot(classifier, accuracy, color='#fc3b00', marker='o')
# plt.title('Accuracy  rate using Bagging and Boosting Model Classifier', fontsize=14)
# # plt.xlabel('Classifier', fontsize=14)
# plt.ylabel('Accuracy(%)', fontsize=14)
# # plt.grid(True)

# xmin, xmax = plt.xlim()
# ymin, ymax = plt.ylim()

# # plt.xlim(-0.5, xmax+0.5)
# plt.ylim(0, ymax * 1.5)
# plt.show()






import numpy as np 
# import matplotlib.pyplot as plt 
  
# labels = ['Precision','Recall','F1-Score','Accuracy']
# mlp = [0.79,0.82,0.77,0.80]
# knn = [0.91,0.79,0.82,0.84]
# dt = [0.32,0.40,0.34,0.36]
# rf = [0.72,0.72,0.62,0.69]


# x = np.arange(len(labels))  # the label locations
# width = 0.2  # the width of the bars

# fig, ax = plt.subplots()
# rects1 = ax.bar(x - width/2, mlp, width, label='MLP',color="#1aba75")
# rects2 = ax.bar(x + width/2, knn, width, label='KNN',color="#20d675")
# rects3 = ax.bar(x - width-0.1, dt, width, label='DT',color="#77e0a8")
# rects4 = ax.bar(x + width+0.1, rf, width, label='RF',color="#2e996b")

# # Add some text for labels, title and custom x-axis tick labels, etc.
# # ax.set_ylabel('Accuracy')
# # ax.set_xlabel('Classifier')
# ax.set_title('Test Model')
# ax.set_xticks(x)
# ax.set_xticklabels(labels)
# ax.legend()

# ax.bar_label(rects1, padding=1)
# ax.bar_label(rects2, padding=1)
# ax.bar_label(rects3, padding=1)
# ax.bar_label(rects4, padding=1)
# xmin, xmax = plt.xlim()
# ymin, ymax = plt.ylim()
# # plt.xlim(0, xmax *1.5)
# plt.ylim(0, ymax * 1.5)
# plt.show()








# labels = ['Precision','Recall','F1-Score','Accuracy']
# mlp = [1.00,1.00,1.00,1.00]
# knn = [0.86,0.85,0.84,0.85]
# dt = [1.00,1.00,1.00,1.00]
# rf = [1.00,1.00,1.00,1.00]

# x = np.arange(len(labels))  # the label locations
# width = 0.2  # the width of the bars

# fig, ax = plt.subplots()
# rects1 = ax.bar(x - width/2, mlp, width, label='MLP',color="#00a2ff")
# rects2 = ax.bar(x + width/2, knn, width, label='KNN',color="#2647ff")
# rects3 = ax.bar(x - width-0.1, dt, width, label='DT',color="#8cd5ff")
# rects4 = ax.bar(x + width+0.1, rf, width, label='RF',color="#6185c9")

# # Add some text for labels, title and custom x-axis tick labels, etc.
# ax.set_title('Train Model')
# ax.set_xticks(x)
# ax.set_xticklabels(labels)
# ax.legend()

# ax.bar_label(rects1, padding=1)
# ax.bar_label(rects2, padding=1)
# ax.bar_label(rects3, padding=1)
# ax.bar_label(rects4, padding=1)
# xmin, xmax = plt.xlim()
# ymin, ymax = plt.ylim()
# # plt.xlim(0, xmax *1.5)
# plt.ylim(0, ymax * 1.5)
# plt.show()




# # Add some text for labels, title and custom x-axis tick labels, etc.

 
# # create a dataset
# height = [80.59, 81.25, 84.37]
# bars = ['Classifier', 'Bagging', 'Boosting']
# x_pos = np.arange(len(bars))

# # Create bars with different colors
# plt.bar(x_pos, height,width=0.4, color=['#02CC9A', '#15A0C1', '#507CAB'])

# # Create names on the x-axis
# plt.xticks(x_pos, bars)
# plt.xlabel('Voting Classifier', fontsize=14)
# plt.title('Voting Classifier Comparison')
# plt.ylabel('Accuracy', fontsize=14)
# # plt.legend(height)
# # Show graph
# xmin, xmax = plt.xlim()
# ymin, ymax = plt.ylim()
# # plt.xlim(0, xmax *1.5)
# plt.ylim(0, ymax)
# plt.show()

# classifier = ['Begging','Boosting']
# accuracy = [85.7,87.5]
  
# plt.plot(classifier, accuracy, color='#fc3b00', marker='o')
# plt.title('Accuracy  rate using Bagging and Boosting Model Classifier', fontsize=14)
# # plt.xlabel('Classifier', fontsize=14)
# plt.ylabel('Accuracy(%)', fontsize=14)
# # plt.grid(True)

# xmin, xmax = plt.xlim()
# ymin, ymax = plt.ylim()

# # plt.xlim(-0.5, xmax+0.5)
# plt.ylim(0, ymax * 1.5)
# plt.show()



import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
 
# Data
labels = ['MLP','KNN','DT','RF']
bg = [78.12,56.25,81.25,78.12]
values = [80.00, 85.00, 40.00, 65.00]

# knn = [0.86,0.85,0.84,0.85]
df=pd.DataFrame({'x_values': labels, 'Bagging Classifier': bg, 'Normal Classifier': values})
 
# multiple line plots
plt.plot( 'x_values', 'Bagging Classifier', data=df, marker='o', markerfacecolor='blue', markersize=12, color='skyblue', linewidth=4)
plt.plot( 'x_values', 'Normal Classifier', data=df, marker='o',markerfacecolor='teal',markersize=12, color='cyan', linewidth=2)
# plt.plot( 'x_values', 'y3_values', data=df, marker='', color='olive', linewidth=2, linestyle='dashed', label="toto")

plt.title('Comparison')
# show legend
plt.legend()

# show graph
plt.show()
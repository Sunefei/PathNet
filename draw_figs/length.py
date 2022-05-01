import matplotlib.pyplot as plt

font = {'family': 'Times New Roman',
        #          'style': 'italic',
        'weight': 'normal',
        'size': 70,
        }

font2 = {'family': 'Times New Roman',
         #          'style': 'italic',
         'weight': 'bold',
         'size': 70,
         }
xtick_size = 60
ytick_size = 60
plt.figure(figsize=(40, 20))
plt.tick_params(labelsize=ytick_size)
plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
                    wspace=0.4, hspace=0.4)
# plt.subplot(2, 1, 1)
# name_list = ['Cora', 'Pubmed', 'Citeseer', 'Cornell', 'NBA', 'BGP', 'Electronics']
# num_list1 = [69.58, 86.80, 65.72, 91.08, 69.52, 60.37, 53.40]
# num_list2 = [84.29, 87.69, 74.52, 85.41, 65.40, 65.42, 69.70]
# num_list3 = [75.23, 85.49, 74.60, 89.19, 66.35, 62.73, 66.24]
# num_list4 = [85.67, 88.14, 78.71, 90.81, 69.52, 65.72, 76.80]
# x = list(range(len(num_list1)))
total_width, n = 0.6, 3
width = total_width / n

# plt.bar(x, num_list1, width=width, label='PAGG-Max', color='wheat')
# for i in range(len(x)):
#     x[i] = x[i] + width
# plt.bar(x, num_list2, width=width, label='PAGG-Sum', tick_label=name_list, color='lightblue')
# for i in range(len(x)):
#     x[i] = x[i] + width
# plt.bar(x, num_list3, width=width, label='PAGG-Complete', color='salmon')
# for i in range(len(x)):
#     x[i] = x[i] + width
# plt.bar(x, num_list4, width=width, label='PAGG', color='dodgerblue')
# plt.legend(prop=font)
# plt.xticks(fontsize=xtick_size)
# # plt.xticks(range(len(max_std)))
# plt.yticks(fontsize=ytick_size)
# plt.xlabel('(a)Model variants of different path embedding aggregators', font2)
# plt.ylabel('Accuracy', font2)
# plt.ylim(50,95)

# plt.subplot(2, 1, 2)
name_list = ['Cora', 'Pubmed', 'Citeseer', 'Cornell', 'NBA', 'BGP', 'Electronics']
num_list1 = [84.30, 87.87, 78.48, 90.00, 69.89, 64.30, 76.70]
num_list2 = [86.67, 87.91, 78.86, 91.62, 72.79, 65.04, 77.02]
num_list3 = [85.54, 88.92, 78.71, 92.43, 71.00, 65.72, 77.84]
num_list4 = [84.50, 86.93, 77.22, 90.38, 69.10, 64.59, 76.67]
x = list(range(len(num_list1)))

plt.bar(x, num_list1, width=width, label='len-2', color='wheat')
for i in range(len(x)):
    x[i] = x[i] + width
plt.bar(x, num_list2, width=width, label='len-3', tick_label=name_list, color='lightblue')
for i in range(len(x)):
    x[i] = x[i] + width
plt.bar(x, num_list3, width=width, label='len-4', color='salmon')
for i in range(len(x)):
    x[i] = x[i] + width
plt.bar(x, num_list4, width=width, label='len-5', color='dodgerblue')
plt.legend(prop=font)
plt.xticks(fontsize=xtick_size)
plt.yticks(fontsize=ytick_size)
plt.xlabel('Model variants of different path length', font2)
plt.ylabel('Accuracy', font2)
plt.ylim(60,95)
plt.show()
plt.savefig(fname="aggregator.eps", format="eps", pad_inches = 0, bbox_inches='tight')
import codecs
import matplotlib.pyplot as plt

def draw(lists, noise_list, name_list):
    # (light light)light red, light blue, (light light)red, blue
    # color_list = ['black', '#F68080', '#99C2FF', '#F03131', '#005FEE']
    # light light light light red, light light light red, light light red, light red, red
    # color_list = ['black', '#F9B0B0', '#F68080', '#F03131', '#DE1010', '#BE0E0E']
    # color_list = ['black', '#F9B0B0', '#F68080', '#F03131', '#DE1010', 'blue', '#BE0E0E']
    color_list = ['#F68080', '#F03131', '#99C2FF', '#005FEE']
    # color_list = ['black', '#DE1010', '#005FEE']

    plt.figure()
    for i in range(len(lists)):
        plt.plot(noise_list, lists[i], label=name_list[i][:-4], color=color_list[i])
    plt.grid(True, linestyle='--', alpha=0.5)  #默认是True，风格设置为虚线，alpha为透明度
    plt.legend()
    font = {'weight': 'normal',
        'size': 13}
    plt.xlabel('Attack Strength(the number of iterations)', font)
    plt.ylabel('Accuracy', font)
    # plt.xlabel('噪声强度')
    # plt.ylabel('分类准确率')
    # plt.savefig('testrobustacc.png')
    # plt.savefig('trainingloss.png')
    # plt.savefig('vanilla&ST&AT.png')
    # plt.savefig('ST&AT-2.png')
    plt.savefig('ST&AT-0_10-13.png')


# filenames = ['vanilla.log', 'ST-epoch200.log', 'ST-epoch300.log', 'ST-epoch400.log', 'AT-epoch200.log', 'AT-epoch400.log']
# filenames = ['ST-epoch200.log', 'ST-epoch300.log', 'ST-epoch400.log', 'AT-epoch200.log', 'AT-epoch400.log']
# filenames = ['ST-epoch200.log', 'ST-epoch400.log', 'AT-epoch200.log', 'AT-epoch400.log']
filenames = ['ST-epoch200.log']
# filenames = ['ST-epoch200.log', 'ST-epoch300.log', 'ST.log', 'AT.log']
# filenames = ['vanilla.log', 'ST.log', 'AT.log']
# filenames = ['noise(Q5-L0.05-R7.65)-cWei-SGD(cosine)-lr0.01-epoch200.log', 'noise(Q5-L0.2-R7.65)-cWei-SGD(cosine)-lr0.01-epoch200.log']
lists = []
noise_list = []
for filename in filenames: 
    f = codecs.open(filename, mode='r', encoding='utf-8')
    for i in range(3):
        line = f.readline()
    alist = []
    count = 0
    while line:
        # # training loss
        # alist.append(float(line.split()[7]))    
        # # testrobustacc
        # if len(line.split())<13:
        #     alist.append(0 if alist==[] else alist[-1])
        # else:
        #     alist.append(float(line.split()[12])) 

        # evaluate result
        alist.append(float(line.split()[7]))
        if filename == filenames[0]:
            noise_list.append(float('{:.2f}'.format(float(line.split()[3]))))

        count += 1
        line = f.readline()
        # if count >= 11:
        #     break
    lists.append(alist)
    f.close()
draw(lists, noise_list, filenames)

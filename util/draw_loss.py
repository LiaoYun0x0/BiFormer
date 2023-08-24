from turtle import color
from matplotlib import rcParams
import matplotlib.pyplot as plt
import re



##读取log文件
logFile = r'artifacts/train_channel/resnet101-dual_softmax_dim256-128_depth256-128/train_SAR2RGB_rotate_mean_std_mish_lbl.log' 
text = ''
file = open(logFile)
for line in file:
    text += line
file.close()

# 

all_list = re.findall('"train_losses": .*[0-9]',text)  
train_losses=[]
train_coarse_loss=[]
val_losses=[]
nums=0
for per_list in all_list:
    # if nums>100:
    #     break
    cur_list = per_list.split(",")
    
    train_losses.append(float(cur_list[0].split(":")[1]))
    val_losses.append(float(cur_list[3].split(":")[1]))
    nums+=1.

print(min(train_losses))

plt.plot(train_losses,label="train_losses")
plt.plot(val_losses,label="test_losses",color="green")
plt.legend()
plt.savefig('train_log_pic/train_channel.png') 

# print(all_list[0].split(","))




# 
# train_acc = []
# for i in all_list:
#     train_acc.append(float(i.split('- accuracy:')[1].split('- val_loss:')[0]))

# val_loss = []
# for i in all_list:
#     val_loss.append(float(i.split('- val_loss:')[1].split('- val_accuracy:')[0]))
    
# val_acc = []
# for i in all_list:
#     val_acc.append(float(i.split('- val_accuracy:')[1]))


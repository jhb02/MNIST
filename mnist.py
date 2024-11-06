import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import matplotlib.pyplot as plt


class Net(torch.nn.Module):

    # 创建神经网络
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.fc1 = torch.nn.Linear(28 * 28, 64)  # 输入为28 * 28像素
        self.fc2 = torch.nn.Linear(64, 64)
        self.fc3 = torch.nn.Linear(64, 64)
        self.fc4 = torch.nn.Linear(64, 10)  # 中间三层都是64个节点，最后输出是10个类别

    def forward(self, x):
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = torch.nn.functional.relu(self.fc3(x))  # x是图像输入，先进行全链接输入计算，再加上一个激活函数，三个输入层
        x = torch.nn.functional.log_softmax(self.fc4(x), dim=1)  # 输出层使用softmax归一化，用log 提高计算准确性，dim = 1,表示沿着类似维度做归一化
        return x


# 导入数据
def get_data_loader(is_train):
    to_tensor = transforms.Compose([transforms.ToTensor()])  # 定义数据转换类型totensor， 张量形式，用于计算多维的数据
    data_set = datasets.MNIST('', is_train, transform=to_tensor, download=True)  # （下载目录， 导入测试集还是训练集， ）
    return DataLoader(data_set, batch_size=15, shuffle=True)  # shuffle = True 表示数据是随机打乱的


# 评估识别正确率
def evaluate(test_data, net):
    n_correct = 0
    n_total = 0
    with torch.no_grad():
        for (x, y) in test_data:  # 按批次取出数据
            outputs = net.forward(x.view(-1, 28 * 28))  # 计算神经网络预测值
            for i, outputs in enumerate(outputs):  # 对批次中每个数据进行比较
                if torch.argmax(outputs) == y[i]:  # argmax计算一个数列中最大值的序号，即预测结果
                    n_correct += 1  # 累加正确的数量
                n_total += 1
    return n_correct / n_total


# 主函数

def main():
    # 导入训练集和测试集,初始化神经网络
    train_data = get_data_loader(is_train=True)
    test_data = get_data_loader(is_train=False)
    net = Net()
    # 打印初始网络正确率
    print('初始网络正确率:', evaluate(test_data, net))
    # 训练神经网络
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    for epoch in range(2):  # epoch是训练的轮次，每个epoch是一个伦茨
        for (x, y) in train_data:
            net.zero_grad()  # 初始化
            output = net.forward(x.view(-1, 28 * 28))  # 正向传播
            loss = torch.nn.functional.nll_loss(output, y)  # 计算差值，nll_loss是对数损失函数，匹配log_softmax中的对数运算
            loss.backward()  # 反向误差传播
            optimizer.step()  # 优化网络参数
        print('epoch:', epoch, 'accuracy:', evaluate(test_data, net))
    for (n, (x, _)) in enumerate(test_data):  # 对神经网络进行检验， 抽取三个图片进行测试
        if n > 3:
            break
        predict = torch.argmax(net.forward(x[0].view(-1, 28 * 28)))
        plt.figure(n)  # 创建一个新的图形窗口，编号为n,可以在同一个脚本显示多个图形
        plt.imshow(x[0].view(28, 28))  # x[0]是第一个样本，view(28, 28) 是将样本形状转换为28*28的二维数组
        plt.title('prediction:' + str(int(predict)))
    plt.show()


if __name__ == '__main__':
    main()  # 如果脚本是主程序调用main函数执行

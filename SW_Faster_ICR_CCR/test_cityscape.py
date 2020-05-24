import os

net = "vgg16"
part = "test_t"
start_epoch = 6
max_epochs = 12
output_dir = "/data/experiments/SW_Faster_ICR_CCR/cityscape/result"
dataset = "cityscape"

for i in range(start_epoch, max_epochs + 1):
    model_dir = "/data/experiments/SW_Faster_ICR_CCR/cityscape/model/cityscape_{}.pth".format(
        i
    )
    command = "python eval/test_SW_ICR_CCR.py --cuda --gc --lc --part {} --net {} --dataset {} --model_dir {} --output_dir {} --num_epoch {}".format(
        part, net, dataset, model_dir, output_dir, i
    )
    os.system(command)

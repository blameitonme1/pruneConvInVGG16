import torch
from dataset import *
from finetune import *
import csv
if __name__ == "__main__":
    # use already prunned model
    model = torch.load("model", map_location=lambda storage, loc : storage)
    model = model.cuda() if torch.cuda.is_available() else model
    loader = test_loader("D:\\DeepLearning\\data\\dogCatClassifier\\test1")
    # 预测
    predictions = []
    correct_predictions = 0
    total_samples = 0
    model.eval()  # 将模型切换到评估模式
    with torch.no_grad():
        for i, (batch, label) in enumerate(loader):
            batch = batch.cuda() if torch.cuda.is_available() else batch
            label = label.cuda() if torch.cuda.is_available() else label
            outputs = model(batch)
            _, predicted = torch.max(outputs, 1)
            predictions.extend(predicted.cpu().numpy())
            correct_predictions += (predicted == label).sum().item()
            total_samples += label.size(0)

    accuracy = correct_predictions / total_samples
    print(f'Accuracy: {accuracy * 100:.2f}%')

    # 保存预测结果到 CSV 文件
    with open('submissions.csv', 'w', newline='') as csvfile:
        fieldnames = ['id', 'label']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for i, label in enumerate(predictions):
            writer.writerow({'id': i + 1, 'label': label})
import torch
import os
from PIL import Image
from finetune import *
import csv
class CustomTestDataset(data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = [os.path.join(root_dir, img) for img in os.listdir(root_dir)]
        # print(self.image_paths)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path)
        img = self.transform(img)
        label = img_path.split('\\')[-1].split('.')[0]
        # print(img_path)
        print(label)
        if label == 'dog':
            label=1
        elif label == 'cat':
            label=0
        return img, label
def pred_loader(path, batch_size=64, num_workers=4, pin_memory=True):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    test_set = CustomTestDataset(path, transform=transform)

    return data.DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
def test_model(model, test_data_path, batch_size=32, num_workers=4, pin_memory=True):
    # 定义图像预处理
    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

    # 创建测试数据集
    test_dataset = datasets.ImageFolder(test_data_path, transform=transform)

    # 创建数据加载器
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    # 模型评估
    model.eval()
    correct_predictions = 0
    total_samples = 0
    with torch.no_grad():
        for i, (batch, labels) in enumerate(test_loader):
            batch = batch.cuda() if torch.cuda.is_available() else batch
            labels = labels.cuda() if torch.cuda.is_available() else labels
            outputs = model(batch)
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_samples += labels.size(0)

    accuracy = correct_predictions / total_samples
    print(f'Accuracy: {accuracy * 100:.2f}%')

    return accuracy

if __name__ == "__main__":
    # use already prunned model
    model = torch.load("model", map_location=lambda storage, loc : storage)
    # model = ModifiedVGG16Model()
    model = model.cuda() if torch.cuda.is_available() else model
    test_model(model, "D:\\DeepLearning\\data\\dogCatClassifier\\test0")
    loader = pred_loader("D:\\DeepLearning\\data\\dogCatClassifier\\test1\\test1\\test1")
    # 预测
    # model.eval()
    # probabilities = []
    # with torch.no_grad():
    #     for i, (batch, _) in enumerate(loader):
    #         batch = batch.cuda() if torch.cuda.is_available() else batch
    #         outputs = model(batch)
    #         probabilities.extend(torch.nn.functional.softmax(outputs, dim=1)[:, 1].cpu().numpy())

    # # 保存预测结果到 CSV 文件
    # with open('submissions.csv', 'w', newline='') as csvfile:
    #     fieldnames = ['id', 'label']
    #     writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    #     writer.writeheader()
    #     for i, prob in enumerate(probabilities):
    #         writer.writerow({'id': i + 1, 'label': prob})
    probabilities = []
    with torch.no_grad():
        for i, (batch, fileid) in enumerate(loader):
            # print(image_names)
            batch = batch.cuda() if torch.cuda.is_available() else batch
            outputs = model(batch)
            preds_list = torch.nn.functional.softmax(outputs, dim=1)[:, 1].tolist()
            probabilities += list(zip(list(fileid), preds_list))
    
    # probabilities.sort(key = lambda x : int(x[0]))
    # 保存预测结果到 CSV 文件
    probabilities.sort(key=lambda x: int(x[0]))  # Sort by fileid

    # 保存预测结果到 CSV 文件
    with open('submissions.csv', 'w', newline='') as csvfile:
        fieldnames = ['id', 'label']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for i, (fileid, label) in enumerate(probabilities):
            #writer.writerow({'id': i + 1, 'label': label})
            writer.writerow({'id': fileid, 'label': label})
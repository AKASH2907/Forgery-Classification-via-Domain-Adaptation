import torch
import os
import math
import data_loader
import models
from config import CFG
import utils
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

MODEL_FILE = './trained_models/alex_cmf10.pt'

def test(model, target_test_loader):
    model.eval()
    test_loss = utils.AverageMeter()
    correct = 0
    accs = 0
    f1s = 0
    best_accuracy = 0
    new_f1 = 0
    criterion = torch.nn.CrossEntropyLoss()
    len_target_dataset = len(target_test_loader.dataset)
    print(len_target_dataset)
    len_target = len_target_dataset/128
    with torch.no_grad():
        for data, target in target_test_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            s_output = model.predict(data)
            # print(s_output)
            loss = criterion(s_output, target)

            test_loss.update(loss.item())
            pred = torch.max(s_output, 1)[1]

            preds = pred.cpu().detach().numpy()
            targets = target.cpu().detach().numpy()
            
            accs += accuracy_score(preds, targets)
            f1s += f1_score(preds, targets)
            correct += torch.sum(pred == target)
    new_f1 = 100. * f1s / len_target
    print(new_f1, best_accuracy)
    if new_f1>best_accuracy:
        print("saving model...")
        best_accuracy = new_f1
        torch.save(model.state_dict(), 'trained_models/alex_cmf10.pt')

    print('{} --> {}: max correct: {}, accuracy{: .2f}%\n'.format(
        source_name, target_name, correct, 100. * correct / len_target_dataset))
    print("accuracy_score:", 100. * accs / len_target)
    print("f1_score:", 100. * f1s / len_target)

def model_evaluate(model, target_test_loader):
    model.load_state_dict(torch.load(MODEL_FILE))
    print("Model Loaded...")
    model.eval()
    correct = 0
    accs = 0
    f1s = 0
    len_target_dataset = len(target_test_loader.dataset)
    # print(len_target_dataset)
    len_target = len_target_dataset/128
    with torch.no_grad():
        for data, target in target_test_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            s_output = model.predict(data)
            pred = torch.max(s_output, 1)[1]

            preds = pred.cpu().detach().numpy()
            targets = target.cpu().detach().numpy()
            
            accs += accuracy_score(preds, targets)
            f1s += f1_score(preds, targets)
            correct += torch.sum(pred == target)

    # print('{} --> {}: max correct: {}, accuracy{: .2f}%\n'.format(
    #     source_name, target_name, correct, 100. * correct / len_target_dataset))
    print("accuracy_score:", 100. * accs / len_target)
    print("f1_score:", 100. * f1s / len_target)

def train(source_loader, target_train_loader, target_test_loader, model, optimizer, CFG):
    len_source_loader = len(source_loader)
    len_target_loader = len(target_train_loader)
    train_loss_clf = utils.AverageMeter()
    train_loss_transfer = utils.AverageMeter()
    train_loss_total = utils.AverageMeter()
    for e in range(CFG['epoch']):
        model.train()
        iter_source, iter_target = iter(
            source_loader), iter(target_train_loader)
        n_batch = min(len_source_loader, len_target_loader)
        criterion = torch.nn.CrossEntropyLoss()
        for i in range(n_batch):
            data_source, label_source = iter_source.next()
            data_target, _ = iter_target.next()
            data_source, label_source = data_source.to(
                DEVICE), label_source.to(DEVICE)
            data_target = data_target.to(DEVICE)

            optimizer.zero_grad()
            label_source_pred, transfer_loss = model(data_source, data_target)
            clf_loss = criterion(label_source_pred, label_source)
            loss = clf_loss + CFG['lambda'] * transfer_loss
            loss.backward()
            optimizer.step()
            train_loss_clf.update(clf_loss.item())
            train_loss_transfer.update(transfer_loss.item())
            train_loss_total.update(loss.item())
            if i % CFG['log_interval'] == 0:
                print('Train Epoch: [{}/{} ({:02d}%)], cls_Loss: {:.6f}, transfer_loss: {:.6f}, total_Loss: {:.6f}'.format(
                    e + 1,
                    CFG['epoch'],
                    int(100. * i / n_batch), train_loss_clf.avg, train_loss_transfer.avg, train_loss_total.avg))

        # Test
        test(model, target_train_loader)

    test(model, target_test_loader)


def load_data(src, tar, root_dir):
    # folder_src = root_dir + src + '/images/'
    # folder_tar = root_dir + tar + '/images/'
    
    training_dir = '../'
    testing_dir_tr = '../cmfd_forge_train/'
    testing_dir_te = '../cmfd_forge_test/'

    transform = transforms.Compose(
        [
         transforms.Resize([224, 224]),
         # transforms.RandomCrop(224),
         transforms.ToTensor(),
         transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
         ]
    )
    data = datasets.ImageFolder(root=os.path.join(training_dir, src), transform=transform)
    shuffled_indices = np.random.permutation(len(data))
    train_idx = shuffled_indices[:int(0.343*len(data))]
    source_loader = DataLoader(data, batch_size=CFG['batch_size'], drop_last=True,
                                # shuffle=True)
                              sampler=SubsetRandomSampler(train_idx),
                              num_workers=4, pin_memory=True)

    # source_loader = data_loader.load_data(
    #     training_dir, src, CFG['batch_size'], True, CFG['kwargs'])
    target_train_loader = data_loader.load_data(
        testing_dir_tr, tar, CFG['batch_size'], True, CFG['kwargs'])
    target_test_loader = data_loader.load_data(
        testing_dir_te, tar, CFG['batch_size'], False, CFG['kwargs'])
    
    print(len(source_loader), len(target_train_loader), len(target_test_loader))

    return source_loader, target_train_loader, target_test_loader

from graphviz import Digraph
import torch
from torch.autograd import Variable


# make_dot was moved to https://github.com/szagoruyko/pytorchviz
from torchviz import make_dot

if __name__ == '__main__':
    torch.manual_seed(0)

    source_name = "cmf"
    target_name = "casia"

    # print('Src: %s, Tar: %s' % (source_name, target_name))

    # source_loader, target_train_loader, target_test_loader = load_data(
    #     source_name, target_name, CFG['data_path'])

    model = models.Transfer_Net(
        CFG['n_class'], transfer_loss='mmd', base_net='alexnet').to(DEVICE)
    optimizer = torch.optim.SGD([
        {'params': model.base_network.parameters()},
        {'params': model.bottleneck_layer.parameters(), 'lr': 10 * CFG['lr']},
        {'params': model.classifier_layer.parameters(), 'lr': 10 * CFG['lr']},
    ], lr=CFG['lr'], momentum=CFG['momentum'], weight_decay=CFG['l2_decay'])

    print(model)
    make_dot(model)
    # train(source_loader, target_train_loader,
    #       target_test_loader, model, optimizer, CFG)

    # model_evaluate(model, target_test_loader)

from tqdm import tqdm
import torch
import torch.optim as optim
import torch.nn.functional as F

def validation(opt, model, validation_dataloader, epoch):
    model.eval()
    val_loss = 0
    correct = 0
    data_iterator_validation = tqdm(validation_dataloader, desc='Epoch {} Validation...'.format(epoch))
    if opt.train_method == 'supervised' and opt.task == 'classification':
        with torch.no_grad():
            for i, (input, class_labels) in enumerate(data_iterator_validation):
                input, class_labels = input.to(opt.device), class_labels.to(opt.device)
                output = model(input)
                predictions = output.argmax(dim=1, keepdims=True).squeeze()
                val_loss += F.nll_loss(output, class_labels, reduction='sum').item()
                correct += (predictions == class_labels).sum().item()
            val_loss /= len(validation_dataloader.dataset)
            accuracy = correct / len(validation_dataloader.dataset)
            print('Epoch {} Validation Results: val_loss {:.2f}, accuracy {:.2f}%'.format(epoch, val_loss, accuracy))
    else:
        raise Exception('No proper setting')

def train(opt, model, train_dataloader, validation_dataloader):
    optimizer = optim.Adadelta(model.parameters(), lr=opt.lr)
    #scheduler = StepLR(optimizer, step_size=1, gamma=opt.gamma)
    model.train()
    for epoch in range(opt.n_epoch):
        data_iterator_training = tqdm(train_dataloader, desc='Epoch {} Training...'.format(epoch))
        
        for i, (input, class_labels) in enumerate(data_iterator_training):
            input, class_labels = input.to(opt.device), class_labels.to(opt.device)
            optimizer.zero_grad()
            output = model(input)
            predictions = output.argmax(dim=1, keepdims=True).squeeze()
            
            # here needs loss function selector
            loss = F.nll_loss(output, class_labels)
            loss.backward()
            optimizer.step()

            correct = (predictions == class_labels).sum().item()
            accuracy = correct / opt.batch_size
            data_iterator_training.set_postfix(loss=loss.item(), accuracy = 100. * accuracy)
        
        validation(opt, model, validation_dataloader, epoch)
        '''    
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
        '''

def test(opt, model, test_dataloader):
    model.eval()
    test_loss = 0
    correct = 0
    data_iterator_test = tqdm(test_dataloader, desc='Testing...')
    with torch.no_grad():
        for i, (input, class_labels) in enumerate(data_iterator_test):
            input, class_labels = input.to(opt.device), class_labels.to(opt.device)
            output = model(input)
            predictions = output.argmax(dim=1, keepdims=True).squeeze()
            test_loss += F.nll_loss(output, class_labels, reduction='sum').item()
            correct += (predictions == class_labels).sum().item()
        test_loss = test_loss / len(test_dataloader.dataset)
        accuracy = correct / len(test_dataloader.dataset)
        print('Test Results: loss {:.2f}, accuracy {:.2f}%'.format(test_loss, accuracy))

    '''
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    '''

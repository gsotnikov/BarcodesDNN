from scheduler import TriangleLR
from argparse import ArgumentParser
from resnet import ResNet, MultipleShallowNets

if __name__ == '__main__':
    parser = ArgumentParser()
    
    parser.add_argument('--type', type=int, help='Type of minima according to the paper: (1 - no scheduling, small LR, 2 - scheduling, with initial high LR)')
    parser.add_argument('--dataset', type=str, help='On which dataset to train ResNet')
    parser.add_argument('--N', type=int, default=8, help='Number of independent models to train')
    parser.add_argument('--device', type=str, default='cuda', help='Device to perform computation')
    
    args = parser.parse_args()
    
    logs = f'Minimas_Dataset_{args.dataset}_Type{args.type}'
    os.makedirs(logs, exist_ok=True)

    
    # Predefined hyperparameters
    N = args.N
    dataset = args.dataset
    device = args.device
    batch_size = 256

    train_loader, test_loader = make_loaders(26, batch_size=batch_size, dataset=dataset)
    batches_in_epoch = len(train_loader)
    best_loss_train = 100

    # No scheduling, small and constant learning rate
    if args.type == 1:
        epochs = 2001
        lr = 1e-4
        knots= [0, 1, 2]
        vals= [lr, lr, lr]
        
    # Scheduling with a starting high learning rate
    elif args.type == 2:
        epochs = 101
        lr = 5e-2
        knots= [0, 10, 50]
        vals= [lr, lr, 1e-4] # First 10 epochs LR is constant, later it is being scheduled to 1e-4
    
    net = MultipleShallowNets([ResNet(dataset, mode='resnet9', seed=26+index) for index in range(N)], kernel_size=3).to(device)
    
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(net.parameters(), lr=0, momentum=0.9, weight_decay=0)
    
    # Obtain a minimum via Outlined procedure
    tlr = TriangleLR(optimizer, batches_in_epoch, knots, vals)
    df = []
    for epoch in range(epochs):

        train_loss = train_scheduler(net, train_loader, optimizer, tlr, epoch, device, dataset=dataset)

        if train_loss < best_loss_train:
            best_loss_train =train_loss

            path = f'{logs}/Best_ResNet9_reg0.torch'
            torch.save(net.state_dict(), path)

        if epoch % 5 == 0:
            test_loss, test_acc = evaluate(net, test_loader, device, dataset=dataset)

            print('Epoch', epoch + 1, 'Train loss', train_loss/N, 'Test loss', test_loss/N, 'Test acc', test_acc)

            df.append([epoch, train_loss/N, test_loss/N, test_acc])

            df_ = pd.DataFrame(df)
            df_.columns = ['epoch', 'train_loss', 'test_loss', 'test_acc']

            df_.to_csv(f'{logs}/ResNet9_reg0.csv', index=False)
        
    path = logs + f'Converged_ResNet9_reg0.torch'
    torch.save(net.state_dict(), path)
    print('NET CONVERGED', 'Seed:', seed, 'Best loss:', best_loss_train/N)
    

    # Make separate
    models_path = f'{logs}/Models/'
    if not os.path.isdir(models_path):
        os.mkdir(models_path)

    net = MultipleShallowNets([ResNet(dataset, mode='resnet9', seed=26+index) for index in range(N)], kernel_size=3).to(device)
    net.load_state_dict(torch.load(f'{logs}/ResNet9_reg0.torch', map_location=device))

    W = factorize_net(net)
    for i in range(N):
        model = ResNet(dataset, mode='resnet9', seed=26)
        model.from_vector(W[:, i])
        torch.save(model.state_dict(), models_path + f'model{i}.torch')
import tqdm
import torch

from data import FastTensorDataLoader, load_data

    
visualize_loss = False
n_epochs = 10
if __name__ == '__main__':
    print('start')
    torch.set_default_dtype(torch.float64)
    model = torch.nn.Sequential(
        torch.nn.Linear(3, 50),
        torch.nn.ReLU(),
        torch.nn.Linear(50, 50),
        torch.nn.ReLU(),
        torch.nn.Linear(50, 3),
    )
    optim = torch.optim.Adam(model.parameters())
    criterion = torch.nn.MSELoss()

    data, meta = load_data()
    dataloader = FastTensorDataLoader(data, dataset_len=100000, batch_size=32, timesteps=2, shuffle=True, pre=True)  # use pre=False if it takes too much memrory to load

    running_losses = []
    for epoch in range(n_epochs):
        running_loss, lossc = 0, 0
        for batch in tqdm.tqdm(dataloader):
            previous, nextt = batch[:, :-1].squeeze(), batch[:, -1]

            optim.zero_grad()
            output = model(previous)
            loss = criterion(output, nextt)
            loss.backward()
            optim.step()

            running_loss = running_loss * lossc / (lossc + 1) + loss.item() / (lossc + 1)
            lossc += 1

        running_losses.append(running_loss)
        print(f"Epoch {epoch}, Mean loss={loss.item()}")
    

    if visualize_loss:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(running_losses)
        plt.show()

    torch.save(model, 'model.checkpoint')


import numpy as np
from timeit import default_timer as timer
import torch.nn as nn
import torch
from torch import optim
import pandas as pd


class trainer():
    
    def __init__(self, model, params):
        """
        
        """
        # Set model.
        self.model = model
        
        # Set training parameters.
        self.num_epochs = params[0]
        assert(self.num_epochs >= 1)
        self.max_epochs_stop = params[1]
        self.num_classes = params[2]
        self.batch_size = params[3]
        self.learning_rate = params[4]
        self.print_every = params[5]
        
        # Set validation stopping variables.
        self.epochs_no_improve = 0
        self.valid_loss_min = np.Inf
        self.valid_max_acc = 0
        self.history = []
        
    
    def test(self):
    
        test_loss, test_acc = 0.0, 0.0

        for data, target in self.model.test_loader:
            try:
                data, target = data.cuda(), target.cuda()
            except:
                print('Can\'t train on CUDA - not available!')

            output = self.model.model(data)

            loss = self.model.criterion(output, target)  
            test_loss += loss.item() * data.size(0)

            _, pred = torch.max(output, dim=1)
            correct_tensor = pred.eq(target.data.view_as(pred))

            accuracy = torch.mean(correct_tensor.type(torch.FloatTensor))
            test_acc += accuracy.item() * data.size(0)

        test_loss = test_loss / len(self.model.test_loader.dataset)
        test_acc = test_acc / len(self.model.test_loader.dataset)
        
        print(f'\nTest Loss: {test_loss} \tTest Accuracy: {test_acc:.4f}')
        return test_loss, test_acc

    
    def train(self):
        """
        
        """
        # Show number of epochs already trained if using loaded in model weights.
        try:
            print(f'Model has been trained for: {self.model.epochs} epochs.\n')
        except:
            self.model.model.epochs = 0
            print(f'Starting Training from Scratch.\n')

        overall_start = timer()
        best_epoch = None
        overall_best_epoch = None

        for epoch in range(self.num_epochs):

            train_loss, valid_loss = 0.0, 0.0
            train_acc, valid_acc = 0, 0

            self.model.model.train()
            start = timer()

            for ii, (data, target) in enumerate(self.model.train_loader):
                try:
                    data, target = data.cuda(), target.cuda()
                except:
                    print('Can\'t train on CUDA - not available!')

                #data, target = data.cuda(), target.cuda()

                self.model.optimizer.zero_grad()
                output = self.model.model(data)

                loss = self.model.criterion(output, target)
                loss.backward()

                self.model.optimizer.step()

                train_loss += loss.item() * data.size(0)

                _, pred = torch.max(output, dim=1)
                correct_tensor = pred.eq(target.data.view_as(pred))
                accuracy = torch.mean(correct_tensor.type(torch.FloatTensor))
                train_acc += accuracy.item() * data.size(0)

                print(
                    f'Epoch: {epoch}\t{100 * (ii + 1) / len(self.model.train_loader):.2f}% complete. {timer() - start:.2f} seconds elapsed in epoch.',
                    end='\r')

            self.model.model.epochs += 1

            with torch.no_grad():
                self.model.model.eval()

                for data, target in self.model.valid_loader:
                    try:
                        data, target = data.cuda(), target.cuda()
                    except:
                        print('Tried CUDA - didn\'t work!')
                        pass

                    output = self.model.model(data)

                    loss = self.model.criterion(output, target)
                    valid_loss += loss.item() * data.size(0)

                    _, pred = torch.max(output, dim=1)
                    correct_tensor = pred.eq(target.data.view_as(pred))
                    accuracy = torch.mean(
                        correct_tensor.type(torch.FloatTensor))
                    valid_acc += accuracy.item() * data.size(0)

                train_loss = train_loss / len(self.model.train_loader.dataset)
                valid_loss = valid_loss / len(self.model.valid_loader.dataset)

                train_acc = train_acc / len(self.model.train_loader.dataset)
                valid_acc = valid_acc / len(self.model.valid_loader.dataset)

                #self.history.append([train_loss, valid_loss, train_acc, valid_acc])

                if (epoch + 1) % self.print_every == 0:
                    print(
                        f'\nEpoch: {epoch} \tTraining Loss: {train_loss:.4f} \tValidation Loss: {valid_loss:.4f}'
                    )
                    print(
                        f'\t\tTraining Accuracy: {100 * train_acc:.2f}%\t Validation Accuracy: {100 * valid_acc:.2f}%'
                    )

                if valid_loss < self.valid_loss_min:
                    torch.save(self.model.model.state_dict(), '_model.ckpt')
                    self.epochs_no_improve = 0
                    self.valid_loss_min = valid_loss
                    valid_best_acc = valid_acc
                    best_epoch = epoch
                    overall_best_epoch = best_epoch

                else:
                    self.epochs_no_improve += 1
                    if self.epochs_no_improve >= self.max_epochs_stop:
                        # print(
                        #     f'\nEarly Stopping! Total epochs: {epoch}. Best epoch: {best_epoch} with loss: {self.valid_loss_min:.2f} and acc: {100 * valid_acc:.2f}%'
                        # )
                        total_time = timer() - overall_start
                        print(
                            f'{total_time:.2f} total seconds elapsed. {total_time / (epoch+1):.2f} seconds per epoch.'
                        )

                        # Load the best state dict
                        #self.model.model.load_state_dict(torch.load("sa))
                        # Attach the optimizer
                        #self.model.optimizer = optimizer

#                         self.history = pd.DataFrame(
#                             self.history,
#                             columns=[
#                                 'train_loss', 'valid_loss', 'train_acc',
#                                  'valid_acc'
#                             ])


        self.model.model.optimizer = self.model.optimizer
        total_time = timer() - overall_start
        # print(
        #     f'\nBest epoch: {overall_best_epoch} with loss: {self.valid_loss_min:.2f} and acc: {100 * valid_acc:.2f}%'
        # )
        print(
            f'{total_time:.2f} total seconds elapsed. {total_time / (epoch+1):.2f} seconds per epoch.'
        )
#         self.history = pd.DataFrame(
#             self.history,
#             columns=['train_loss', 'valid_loss', 'train_acc', 'valid_acc'])
import torch
from network import ConvTasNet
from uPIT_Loss import uPIT
import matplotlib.pyplot as plt
import soundfile as sf
import yaml
import tensorboardX as tbx
import os
from tqdm import tqdm


class Trainer():
    def __init__(self,model,config,time):
        self.model = model
        self.cur_epoch = 0
        self.name = config['name']
        self.C = config['network']['C']
        self.config = config
        self.dir_save = os.path.join('./checkpoint',self.name,time)
        os.makedirs(self.dir_save)

        # setting about optimizer
        opt_name = config['optim']['name']
        weight_decay = config['optim']['weight_decay']
        lr = config['optim']['lr']
        momentum = config['optim']['momentum']

        optimizer = getattr(torch.optim, opt_name)
        if opt_name == 'Adam':
            self.optimizer = optimizer(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            self.optimizer = optimizer(self.model.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum)
        self.clip_norm = config['optim']['clip_norm'] if config['optim']['clip_norm'] else 0
        # setting about machine
        self.device = torch.device(config['gpu'])
        self.parallel = config['parallel']
        if config['training']['resume']['state']:    
            self.load_checkpoint(config)

        self.model = self.model.to(self.device)
        self.model = torch.nn.DataParallel(self.model, device_ids=[0,1,2])
        self.total_epoch = config['training']['total_epoch']
        self.early_stop = config['training']['early_stop']

    def train(self, epoch, dataloader):
        self.model.train()
        num_batchs = len(dataloader)
        total_loss = 0
        for mix, s in tqdm(dataloader):
            mix = mix.to(self.device).detach()
            s = s.to(self.device).detach()
            est_s = self.model(mix)
            epoch_loss = uPIT(est_s,s)

            self.optimizer.zero_grad()
            epoch_loss.backward()
            if self.clip_norm:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(),self.clip_norm)
            self.optimizer.step()
            total_loss += epoch_loss.detach()

        total_loss = total_loss/num_batchs

        return total_loss

    def validation(self, dataloader):
        self.model.eval()
        num_batchs = len(dataloader)
        total_loss = 0
        with torch.no_grad():
            for mix, s in tqdm(dataloader):
                mix = mix.to(self.device)
                s = s.to(self.device)
                est_s = self.model(mix)
                total_loss += uPIT(est_s,s)
        return total_loss/num_batchs

    def est_test(self,mixture):
        est_s = self.model(mix)
        print(est_s.shape)


    
    def run(self,train_dataloader,valid_dataloader):
        train_loss = []
        val_loss = []
        print('cur_epoch',self.cur_epoch)

        writer = tbx.SummaryWriter(self.dir_save)
        self.save_checkpoint(self.cur_epoch,best=False)
        v_loss = self.validation(valid_dataloader)
        best_loss = 1e10
        no_improve = 0
        # starting training part
        while self.cur_epoch < self.total_epoch:
            self.cur_epoch += 1
            t_loss = self.train(self.cur_epoch, train_dataloader)
            print('epoch{0}:train_loss{1}'.format(self.cur_epoch,t_loss))
            v_loss = self.validation(valid_dataloader)
            print('epoch{0}:valid_loss{1}'.format(self.cur_epoch,v_loss))

            writer.add_scalar('t_loss', t_loss, self.cur_epoch)
            writer.add_scalar('v_loss', v_loss, self.cur_epoch)

            if v_loss >= best_loss:
                no_improve += 1
            else:
                best_loss = v_loss
                no_improve = 0
                self.save_checkpoint(self.cur_epoch,best=True)
            
            if no_improve == self.early_stop:
                break
            self.save_checkpoint(self.cur_epoch,best=False)
        
        writer.close()
        

    
    def save_checkpoint(self, epoch, best=True):
        self.model.to('cpu')
        print('save model epoch:{0} as {1}'.format(epoch,"best" if best else "last"))
        path_save_model = os.path.join(self.dir_save,'{0}.pt'.format('best' if best else 'last'))

        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.module.state_dict(),
            'optim_state_dict': self.optimizer.state_dict()
        },
        path_save_model)

        self.model.to(self.device)

        with open(os.path.join(self.dir_save,'config_backup.yaml'),mode='w') as f:
            f.write(yaml.dump(self.config))


    def load_checkpoint(self,config):
        print('load on:',self.device)

        ckp = torch.load(config['training']['resume']['path'],map_location=torch.device('cpu'))
        self.cur_epoch = ckp['epoch']
        self.model.load_state_dict(ckp['model_state_dict'])
        self.optimizer.load_state_dict(ckp['optim_state_dict'])

        self.model = self.model.to(self.device)
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(self.device)

        print('training resume epoch:',self.cur_epoch)

if __name__ == "__main__":
    with open('./config.yaml', 'r') as yml:
        config = yaml.safe_load(yml)

    s1,_ = sf.read("/data1/h_munakata/wsj0/2speakers/min/tr/s1/01aa010b_0.97482_209a010p_-0.97482.wav")
    s2,_ = sf.read("/data1/h_munakata/wsj0/2speakers/min/tr/s2/01aa010b_0.97482_209a010p_-0.97482.wav")
    
    s1 = torch.tensor(s1,dtype=torch.float32).view([1,-1])
    s2 = torch.tensor(s2,dtype=torch.float32).view([1,-1])

    s = torch.cat([s1.view([1,-1,1]), s2.view([1,-1,1])],dim=2)
    mixture = s1+s2

    ctn = ConvTasNet(config)

    optimizer = torch.optim.Adam(ctn.parameters(),lr=0.001)

    for i in tqdm(range(100)):
        est_s = ctn(mixture)
        loss = uPIT(est_s, s)
        print(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    plt.subplot(2,3,1)
    plt.plot(s[:,:,0].view(-1).detach().numpy())
    plt.subplot(2,3,2)
    plt.plot(s[:,:,1].view(-1).detach().numpy())

    plt.subplot(2,3,4)
    plt.plot(est_s[:,:,0].view(-1).detach().numpy())
    plt.subplot(2,3,5)
    plt.plot(est_s[:,:,1].view(-1).detach().numpy())

    plt.savefig('./test.png')
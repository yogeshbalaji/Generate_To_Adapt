import torch
import torch.nn as nn
from torch.autograd import Variable, Function
import torch.optim as optim
import torchvision.utils as vutils
import itertools, datetime
import numpy as np
import models
import utils

class GTA(object):

    def __init__(self, opt, nclasses, mean, std, source_trainloader, source_valloader, targetloader):

        self.source_trainloader = source_trainloader
        self.source_valloader = source_valloader
        self.targetloader = targetloader
        self.opt = opt
        self.mean = mean
        self.std = std
        self.best_val = 0
        
        # Defining networks and optimizers
        self.nclasses = nclasses
        self.netG = models._netG(opt, nclasses)
        self.netD = models._netD(opt, nclasses)
        self.netF = models._netF(opt)
        self.netC = models._netC(opt, nclasses)

        # Weight initialization
        self.netG.apply(utils.weights_init)
        self.netD.apply(utils.weights_init)
        self.netF.apply(utils.weights_init)
        self.netC.apply(utils.weights_init)

        # Defining loss criterions
        self.criterion_c = nn.CrossEntropyLoss()
        self.criterion_s = nn.BCELoss()

        if opt.gpu>=0:
            self.netD.cuda()
            self.netG.cuda()
            self.netF.cuda()
            self.netC.cuda()
            self.criterion_c.cuda()
            self.criterion_s.cuda()

        # Defining optimizers
        self.optimizerD = optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
        self.optimizerG = optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
        self.optimizerF = optim.Adam(self.netF.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
        self.optimizerC = optim.Adam(self.netC.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

        # Other variables
        self.real_label_val = 1
        self.fake_label_val = 0

    """
    Validation function
    """
    def validate(self, epoch):
        
        self.netF.eval()
        self.netC.eval()
        total = 0
        correct = 0
    
        # Testing the model
        for i, datas in enumerate(self.source_valloader):
            inputs, labels = datas         
            inputv, labelv = Variable(inputs.cuda(), volatile=True), Variable(labels.cuda()) 

            outC = self.netC(self.netF(inputv))
            _, predicted = torch.max(outC.data, 1)        
            total += labels.size(0)
            correct += ((predicted == labels.cuda()).sum())
            
        val_acc = 100*float(correct)/total
        print('%s| Epoch: %d, Val Accuracy: %f %%' % (datetime.datetime.now(), epoch, val_acc))
    
        # Saving checkpoints
        torch.save(self.netF.state_dict(), '%s/models/netF.pth' %(self.opt.outf))
        torch.save(self.netC.state_dict(), '%s/models/netC.pth' %(self.opt.outf))
        
        if val_acc>self.best_val:
            self.best_val = val_acc
            torch.save(self.netF.state_dict(), '%s/models/model_best_netF.pth' %(self.opt.outf))
            torch.save(self.netC.state_dict(), '%s/models/model_best_netC.pth' %(self.opt.outf))
            
            
    """
    Train function
    """
    def train(self):
        
        curr_iter = 0
        
        reallabel = torch.FloatTensor(self.opt.batchSize).fill_(self.real_label_val)
        fakelabel = torch.FloatTensor(self.opt.batchSize).fill_(self.fake_label_val)
        if self.opt.gpu>=0:
            reallabel, fakelabel = reallabel.cuda(), fakelabel.cuda()
        reallabelv = Variable(reallabel) 
        fakelabelv = Variable(fakelabel) 
        
        for epoch in range(self.opt.nepochs):
            
            self.netG.train()    
            self.netF.train()    
            self.netC.train()    
            self.netD.train()    
        
            for i, (datas, datat) in enumerate(itertools.izip(self.source_trainloader, self.targetloader)):
                
                ###########################
                # Forming input variables
                ###########################
                
                src_inputs, src_labels = datas
                tgt_inputs, __ = datat       
                src_inputs_unnorm = (((src_inputs*self.std[0]) + self.mean[0]) - 0.5)*2

                # Creating one hot vector
                labels_onehot = np.zeros((self.opt.batchSize, self.nclasses+1), dtype=np.float32)
                for num in range(self.opt.batchSize):
                    labels_onehot[num, src_labels[num]] = 1
                src_labels_onehot = torch.from_numpy(labels_onehot)

                labels_onehot = np.zeros((self.opt.batchSize, self.nclasses+1), dtype=np.float32)
                for num in range(self.opt.batchSize):
                    labels_onehot[num, self.nclasses] = 1
                tgt_labels_onehot = torch.from_numpy(labels_onehot)
                
                if self.opt.gpu>=0:
                    src_inputs, src_labels = src_inputs.cuda(), src_labels.cuda()
                    src_inputs_unnorm = src_inputs_unnorm.cuda() 
                    tgt_inputs = tgt_inputs.cuda()
                    src_labels_onehot = src_labels_onehot.cuda()
                    tgt_labels_onehot = tgt_labels_onehot.cuda()
                
                # Wrapping in variable
                src_inputsv, src_labelsv = Variable(src_inputs), Variable(src_labels)
                src_inputs_unnormv = Variable(src_inputs_unnorm)
                tgt_inputsv = Variable(tgt_inputs)
                src_labels_onehotv = Variable(src_labels_onehot)
                tgt_labels_onehotv = Variable(tgt_labels_onehot)
                
                ###########################
                # Updates
                ###########################
                
                # Updating D network
                
                self.netD.zero_grad()
                src_emb = self.netF(src_inputsv)
                src_emb_cat = torch.cat((src_labels_onehotv, src_emb), 1)
                src_gen = self.netG(src_emb_cat)

                tgt_emb = self.netF(tgt_inputsv)
                tgt_emb_cat = torch.cat((tgt_labels_onehotv, tgt_emb),1)
                tgt_gen = self.netG(tgt_emb_cat)

                src_realoutputD_s, src_realoutputD_c = self.netD(src_inputs_unnormv)   
                errD_src_real_s = self.criterion_s(src_realoutputD_s, reallabelv) 
                errD_src_real_c = self.criterion_c(src_realoutputD_c, src_labelsv) 

                src_fakeoutputD_s, src_fakeoutputD_c = self.netD(src_gen)
                errD_src_fake_s = self.criterion_s(src_fakeoutputD_s, fakelabelv)

                tgt_fakeoutputD_s, tgt_fakeoutputD_c = self.netD(tgt_gen)          
                errD_tgt_fake_s = self.criterion_s(tgt_fakeoutputD_s, fakelabelv)

                errD = errD_src_real_c + errD_src_real_s + errD_src_fake_s + errD_tgt_fake_s
                errD.backward(retain_graph=True)    
                self.optimizerD.step()
                

                # Updating G network
                
                self.netG.zero_grad()       
                src_fakeoutputD_s, src_fakeoutputD_c = self.netD(src_gen)
                errG_c = self.criterion_c(src_fakeoutputD_c, src_labelsv)
                errG_s = self.criterion_s(src_fakeoutputD_s, reallabelv)
                errG = errG_c + errG_s
                errG.backward(retain_graph=True)
                self.optimizerG.step()
                

                # Updating C network
                
                self.netC.zero_grad()
                outC = self.netC(src_emb)   
                errC = self.criterion_c(outC, src_labelsv)
                errC.backward(retain_graph=True)    
                self.optimizerC.step()

                
                # Updating F network

                self.netF.zero_grad()
                errF_fromC = self.criterion_c(outC, src_labelsv)        

                src_fakeoutputD_s, src_fakeoutputD_c = self.netD(src_gen)
                errF_src_fromD = self.criterion_c(src_fakeoutputD_c, src_labelsv)*(self.opt.adv_weight)

                tgt_fakeoutputD_s, tgt_fakeoutputD_c = self.netD(tgt_gen)
                errF_tgt_fromD = self.criterion_s(tgt_fakeoutputD_s, reallabelv)*(self.opt.adv_weight*self.opt.alpha)
                
                errF = errF_fromC + errF_src_fromD + errF_tgt_fromD
                errF.backward()
                self.optimizerF.step()        
                
                curr_iter += 1
                
                # Visualization
                if i == 1:
                    vutils.save_image((src_gen.data/2)+0.5, '%s/visualization/source_gen_%d.png' %(self.opt.outf, epoch))
                    vutils.save_image((tgt_gen.data/2)+0.5, '%s/visualization/target_gen_%d.png' %(self.opt.outf, epoch))
                    
                # Learning rate scheduling
                if self.opt.lrd:
                    self.optimizerD = utils.exp_lr_scheduler(self.optimizerD, epoch, self.opt.lr, self.opt.lrd, curr_iter)    
                    self.optimizerF = utils.exp_lr_scheduler(self.optimizerF, epoch, self.opt.lr, self.opt.lrd, curr_iter)
                    self.optimizerC = utils.exp_lr_scheduler(self.optimizerC, epoch, self.opt.lr, self.opt.lrd, curr_iter)                  
            
            # Validate every epoch
            self.validate(epoch+1)


class Sourceonly(object):

    def __init__(self, opt, nclasses, source_trainloader, source_valloader):

        self.source_trainloader = source_trainloader
        self.source_valloader = source_valloader
        self.opt = opt
        self.best_val = 0
        
        # Defining networks and optimizers
        self.nclasses = nclasses
        self.netF = models._netF(opt)
        self.netC = models._netC(opt, nclasses)

        # Weight initialization
        self.netF.apply(utils.weights_init)
        self.netC.apply(utils.weights_init)

        # Defining loss criterions
        self.criterion = nn.CrossEntropyLoss()

        if opt.gpu>=0:
            self.netF.cuda()
            self.netC.cuda()
            self.criterion.cuda()

        # Defining optimizers
        self.optimizerF = optim.Adam(self.netF.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
        self.optimizerC = optim.Adam(self.netC.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))


    """
    Validation function
    """
    def validate(self, epoch):
        
        self.netF.eval()
        self.netC.eval()
        total = 0
        correct = 0
    
        # Testing the model
        for i, datas in enumerate(self.source_valloader):
            inputs, labels = datas         
            inputv, labelv = Variable(inputs.cuda(), volatile=True), Variable(labels.cuda()) 

            outC = self.netC(self.netF(inputv))
            _, predicted = torch.max(outC.data, 1)        
            total += labels.size(0)
            correct += ((predicted == labels.cuda()).sum())
            
        val_acc = 100*float(correct)/total
        print('%s| Epoch: %d, Val Accuracy: %f %%' % (datetime.datetime.now(), epoch, val_acc))
    
        # Saving checkpoints
        torch.save(self.netF.state_dict(), '%s/models/netF_sourceonly.pth' %(self.opt.outf))
        torch.save(self.netC.state_dict(), '%s/models/netC_sourceonly.pth' %(self.opt.outf))
        
        if val_acc>self.best_val:
            self.best_val = val_acc
            torch.save(self.netF.state_dict(), '%s/models/model_best_netF_sourceonly.pth' %(self.opt.outf))
            torch.save(self.netC.state_dict(), '%s/models/model_best_netC_sourceonly.pth' %(self.opt.outf))
            
    
    """
    Train function
    """
    def train(self):
        
        curr_iter = 0
        for epoch in range(self.opt.nepochs):
            
            self.netF.train()    
            self.netC.train()    
        
            for i, datas in enumerate(self.source_trainloader):
                
                ###########################
                # Forming input variables
                ###########################
                
                src_inputs, src_labels = datas
                if self.opt.gpu>=0:
                    src_inputs, src_labels = src_inputs.cuda(), src_labels.cuda()
                src_inputsv, src_labelsv = Variable(src_inputs), Variable(src_labels)
                
                ###########################
                # Updates
                ###########################
                
                self.netC.zero_grad()
                self.netF.zero_grad()
                outC = self.netC(self.netF(src_inputsv))   
                loss = self.criterion(outC, src_labelsv)
                loss.backward()    
                self.optimizerC.step()
                self.optimizerF.step()

                curr_iter += 1
                
                # Learning rate scheduling
                if self.opt.lrd:
                    self.optimizerF = utils.exp_lr_scheduler(self.optimizerF, epoch, self.opt.lr, self.opt.lrd, curr_iter)
                    self.optimizerC = utils.exp_lr_scheduler(self.optimizerC, epoch, self.opt.lr, self.opt.lrd, curr_iter)                  
            
            # Validate every epoch
            self.validate(epoch)

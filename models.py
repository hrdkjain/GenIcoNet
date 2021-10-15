import sys
import torch.nn

sys.path.append('../IcosahedralCNN/')
from icocnn.ico_conv import IcoConvS2S
from icocnn.ico_conv import IcoUpsampleS2S

""" Base Models """
class IcoUpS2S(torch.nn.Module):
    def __init__(self, in_features, out_features, bias=True, subdivisions=0,
                 corner_mode='zeros'):
        super().__init__()
        self.up = IcoUpsampleS2S(in_features, subdivisions, corner_mode)
        self.conv = IcoConvS2S(in_features, out_features, 1, bias, subdivisions+1,
                               corner_mode=corner_mode)

    def forward(self, x):
        x = self.up(x)
        x = self.conv(x)
        return x

class BasicIcoS2SDownBlock(torch.nn.Module):
    def __init__(self, in_features, out_features, bias, in_subdivisions, corner_mode):
        super(BasicIcoS2SDownBlock, self).__init__()
        self.conv00 = IcoConvS2S(in_features=in_features, out_features=out_features, stride=2,
                                 bias=bias, subdivisions=in_subdivisions, corner_mode=corner_mode)
        self.icobn00 = torch.nn.BatchNorm2d(out_features)
        self.conv01 = IcoConvS2S(in_features=out_features, out_features=out_features, stride=1,
                                 bias=bias, subdivisions=in_subdivisions-1, corner_mode=corner_mode)
        self.icobn01 = torch.nn.BatchNorm2d(out_features)

        self.conv10 = IcoConvS2S(in_features=in_features, out_features=out_features, stride=2,
                                 bias=bias, subdivisions=in_subdivisions, corner_mode=corner_mode)
        self.icobn10 = torch.nn.BatchNorm2d(out_features)

    def forward(self, x):
        out0 = self.icobn01(self.conv01(torch.nn.functional.relu(self.icobn00(self.conv00(x)))))
        out1 = self.icobn10(self.conv10(x))
        out = torch.nn.functional.relu(out0+out1)
        return out

class BasicIcoS2SUpBlock(torch.nn.Module):
    def __init__(self, in_features, out_features, bias, in_subdivisions, corner_mode):
        super(BasicIcoS2SUpBlock, self).__init__()
        self.upsample00 = IcoUpsampleS2S(in_features, in_subdivisions, corner_mode)
        self.conv00 = IcoConvS2S(in_features=in_features, out_features=out_features, stride=1,
                                 bias=bias, subdivisions=in_subdivisions+1, corner_mode=corner_mode)
        self.icobn00 = torch.nn.BatchNorm2d(out_features)
        self.conv01 = IcoConvS2S(in_features=out_features, out_features=out_features, stride=1,
                                 bias=bias, subdivisions=in_subdivisions+1, corner_mode=corner_mode)
        self.icobn01 = torch.nn.BatchNorm2d(out_features)

        self.upsample10 = IcoUpsampleS2S(in_features, in_subdivisions, corner_mode)
        self.conv10 = IcoConvS2S(in_features=in_features, out_features=out_features, stride=1,
                                 bias=bias, subdivisions=in_subdivisions+1, corner_mode=corner_mode)
        self.icobn10 = torch.nn.BatchNorm2d(out_features)

    def forward(self, x):
        out0 = self.icobn01(self.conv01(torch.nn.functional.relu(self.icobn00(self.conv00(self.upsample00(x))))))
        out1 = self.icobn10(self.conv10(self.upsample10(x)))
        out = torch.nn.functional.relu(out0+out1)
        return out

class Identity(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.W = torch.nn.Parameter(torch.randn(1, 3, 160, 64))
        self.W.requires_grad = True

    def forward(self, x):
        n = x.size(0)
        t = torch.cat(n*[self.W]) - torch.cat(n*[self.W])
        return x + t

class VAE(torch.nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.encoder = None
        self.mu = None
        self.logvar = None
        self.decoder = None

    def encode(self, input):
        raise NotImplementedError

    def decode(self, input):
        raise NotImplementedError

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return  self.decode(z), mu, logvar


""" Base Layers """
def createico2enc(corner_mode = 'average', model='simple'):
    if model == 'residualS2S':
        encoder = torch.nn.Sequential(
            IcoConvS2S(in_features=3,
                       out_features=64,
                       stride=1,
                       bias=True,
                       subdivisions=5,
                       corner_mode=corner_mode),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(inplace=False),
            BasicIcoS2SDownBlock(in_features=64,
                                 out_features=128,
                                 bias=True,
                                 in_subdivisions=5,
                                 corner_mode=corner_mode),
            BasicIcoS2SDownBlock(in_features=128,
                                 out_features=256,
                                 bias=True,
                                 in_subdivisions=4,
                                 corner_mode=corner_mode),
            BasicIcoS2SDownBlock(in_features=256,
                                 out_features=256,
                                 bias=True,
                                 in_subdivisions=3,
                                 corner_mode=corner_mode),
        )
    elif model == 'identity':
        encoder = torch.nn.Sequential(Identity())
    return encoder

def createenc2ico(corner_mode = 'average', model='simple'):
    if model == 'residualS2S':
        decoder = torch.nn.Sequential(
            BasicIcoS2SUpBlock(in_features=256,
                               out_features=256,
                               bias=True,
                               in_subdivisions=2,
                               corner_mode=corner_mode),
            BasicIcoS2SUpBlock(in_features=256,
                               out_features=128,
                               bias=True,
                               in_subdivisions=3,
                               corner_mode=corner_mode),
            BasicIcoS2SUpBlock(in_features=128,
                               out_features=64,
                               bias=True,
                               in_subdivisions=4,
                               corner_mode=corner_mode)
        )
        enc2icoConv = torch.nn.Sequential(torch.nn.Conv2d(in_channels=64,
                                                          out_channels=3,
                                                          kernel_size=(1, 1)),
                                          torch.nn.Tanh()
                                          )
        return decoder, enc2icoConv
    elif model == 'identity':
        decoder = torch.nn.Sequential(Identity())
        return decoder, torch.nn.Sequential(torch.nn.Identity())


def createico2enc_vae(corner_mode = 'average', model='simple'):
    if model == 'residualS2S':
        encoder = torch.nn.Sequential(
            IcoConvS2S(in_features=3,
                       out_features=64,
                       stride=1,
                       bias=True,
                       subdivisions=5,
                       corner_mode=corner_mode),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(inplace=False),
            BasicIcoS2SDownBlock(in_features=64,
                                 out_features=128,
                                 bias=True,
                                 in_subdivisions=5,
                                 corner_mode=corner_mode),
            BasicIcoS2SDownBlock(in_features=128,
                                 out_features=256,
                                 bias=True,
                                 in_subdivisions=4,
                                 corner_mode=corner_mode),
        )
    elif model == 'identity':
        encoder = torch.nn.Sequential(Identity())
    return encoder

def createenc2ico_vae(corner_mode = 'average', model='simple'):
    if model == 'residualS2S':
        decoder = torch.nn.Sequential(
            BasicIcoS2SUpBlock(in_features=512,
                               out_features=256,
                               bias=True,
                               in_subdivisions=2,
                               corner_mode=corner_mode),
            BasicIcoS2SUpBlock(in_features=256,
                               out_features=128,
                               bias=True,
                               in_subdivisions=3,
                               corner_mode=corner_mode),
            BasicIcoS2SUpBlock(in_features=128,
                               out_features=64,
                               bias=True,
                               in_subdivisions=4,
                               corner_mode=corner_mode)
        )
        enc2icoConv = torch.nn.Sequential(torch.nn.Conv2d(in_channels=64,
                                                          out_channels=3,
                                                          kernel_size=(1, 1)),
                                          torch.nn.Tanh()
                                          )
        return decoder, enc2icoConv

    elif model == 'identity':
        decoder = torch.nn.Sequential(Identity())
        return decoder, torch.nn.Sequential(torch.nn.Identity())

""" Derived Models """
class ico2ico(torch.nn.Module):
    def __init__(self, params):
        super(ico2ico, self).__init__()
        self.encoder = createico2enc(params['ico']['corner_mode'], params['ico2ico']['model'])
        self.enc = torch.nn.Identity()
        self.subdivisions = params['ico']['subdivisions']
        self.decoder, self.enc2icoConv = createenc2ico(params['ico']['corner_mode'], params['ico2ico']['model'])

    def forward(self,x):
        x = self.encoder(x)
        x = self.enc(x)
        x = self.decoder(x)
        x = self.enc2icoConv(x)
        return x

class ico2enc(torch.nn.Module):
    def __init__(self, params):
        super(ico2enc, self).__init__()
        self.encoder = createico2enc(params['ico']['corner_mode'], params['ico2ico']['model'])

    def forward(self, x):
        x = self.encoder(x)
        return x

class enc2ico(torch.nn.Module):
    def __init__(self, params):
        super(enc2ico, self).__init__()
        self.subdivisions = params['ico']['subdivisions']
        self.decoder, self.enc2icoConv = createenc2ico(params['ico']['corner_mode'], params['ico2ico']['model'])

    def forward(self, x):
        x = self.decoder(x)
        x = self.enc2icoConv(x)
        return x

class ico2ico_vae(VAE):
    def __init__(self, params):
        super(ico2ico_vae, self).__init__()
        self.params = params
        self.model = params[params['model_name']]['model']
        self.encoder = createico2enc_vae(params['ico']['corner_mode'], self.model)
        self.mu = self.createMu()
        self.logvar = self.createLogvar()
        self.mu_hook = torch.nn.Identity()
        self.logvar_hook = torch.nn.Identity()
        self.reparameterize_hook = torch.nn.Identity()
        self.subdivisions = params['ico']['subdivisions']
        self.decoder, self.final_layer = createenc2ico_vae(params['ico']['corner_mode'], self.model)

    def createMu(self):
        return torch.nn.Sequential(IcoConvS2S(in_features=256,
                                              out_features=512,
                                              stride=2,
                                              bias=True,
                                              subdivisions=3,
                                              corner_mode=self.params['ico']['corner_mode']),
                                   torch.nn.BatchNorm2d(512)
                                   )

    def createLogvar(self):
        return torch.nn.Sequential(IcoConvS2S(in_features=256,
                                              out_features=512,
                                              stride=2,
                                              bias=True,
                                              subdivisions=3,
                                              corner_mode=self.params['ico']['corner_mode']),
                                   torch.nn.BatchNorm2d(512)
                                   )

    def encode(self, input):
        result = self.encoder(input)
        mu = self.mu(result)
        mu = self.mu_hook(mu)
        logvar = self.logvar(result)
        logvar = self.logvar_hook(logvar)
        return mu, logvar

    def decode(self, z):
        result = self.reparameterize_hook(z)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

class ico2enc_vae(VAE):
    def __init__(self, params):
        super(ico2enc_vae, self).__init__()
        self.params = params
        self.model = params[params['model_name']]['model']
        self.encoder = createico2enc_vae(params['ico']['corner_mode'], self.model)
        self.mu = ico2ico_vae.createMu(self)
        self.logvar = ico2ico_vae.createLogvar(self)

    def encode(self, input):
        result = self.encoder(input)
        mu = self.mu(result)
        logvar = self.logvar(result)
        return mu, logvar

    def forward(self, x):
        mu, logvar = self.encode(x)
        return  mu, logvar

class enc2ico_vae(VAE):
    def __init__(self, params):
        super(enc2ico_vae, self).__init__()
        self.params = params
        self.model = params[params['model_name']]['model']
        self.subdivisions = params['ico']['subdivisions']
        self.decoder, self.final_layer = createenc2ico_vae(params['ico']['corner_mode'], self.model)

    def createSample(self, batch_size, misc):
        trn_mean = misc[0]['trn_mean']
        trn_logvar = misc[0]['trn_logvar']
        return torch.add(trn_mean,trn_logvar*torch.randn(trn_logvar.shape))

    def decode(self, z):
        result = self.decoder(z)
        result = self.final_layer(result)
        return result

    def forward(self, x):
        return self.decode(x), torch.tensor([]), torch.tensor([])

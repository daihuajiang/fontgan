import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler


class UNetGenerator(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self, input_nc=3, output_nc=3, num_downs=8, ngf=64, embedding_num=40, embedding_dim=128,
                 norm_layer=nn.BatchNorm2d, use_dropout=False):
        """
        Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(UNetGenerator, self).__init__()
        # construct unet structure

        unet_block = CrossAttentionBlock(ngf * 8)
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer,
                                             innermost=True, embedding_dim=embedding_dim)  # add the innermost layer
        for _ in range(num_downs - 5):  # add intermediate layers with ngf * 8 filters
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block,
                                                 norm_layer=norm_layer)
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block,
                                             norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block,
                                             norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block,
                                             norm_layer=norm_layer)
        self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block,
                                             outermost=True,
                                             norm_layer=norm_layer)  # add the outermost layer
        self.embedder = nn.Embedding(embedding_num, embedding_dim)
        

    def forward(self, x, style_or_label=None, skel=None):
        """Standard forward"""
        if style_or_label is not None and 'LongTensor' in style_or_label.type():
            return self.model(x, self.embedder(style_or_label), skel)
        else:
            return self.model(x, style_or_label, skel)

class SelfAttentionBlock(nn.Module):
    """ Self attention Layer"""
    def __init__(self,in_dim):
        super(SelfAttentionBlock,self).__init__()
        self.chanel_in = in_dim
        
        self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.key_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.value_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax  = nn.Softmax(dim=-1) #
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature 
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize,C,width ,height = x.size()
        proj_query  = self.query_conv(x).view(m_batchsize,-1,width*height).permute(0,2,1) # B X CX(N)
        proj_key =  self.key_conv(x).view(m_batchsize,-1,width*height) # B X C x (*W*H)
        energy =  torch.bmm(proj_query,proj_key) # transpose check
        attention = self.softmax(energy) # BX (N) X (N) 
        proj_value = self.value_conv(x).view(m_batchsize,-1,width*height) # B X C X N

        out = torch.bmm(proj_value,attention.permute(0,2,1) )
        out = out.view(m_batchsize,C,width,height)
        
        out = self.gamma*out + x
        
        return out

class CrossAttentionBlock(nn.Module):
    """ Cross attention Layer"""
    def __init__(self,in_dim):
        super(CrossAttentionBlock,self).__init__()
        self.chanel_in = in_dim
        
        self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.key_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.value_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax  = nn.Softmax(dim=-1) #
    def forward(self, x, style=None, skel=None):
        """
            inputs :
                x : input feature maps( B X C X W X H)
                skel : input feature maps( B X C X W X H)
            returns :
                out : Cross attention value + input feature x
                attention: B X N X N (N is Width*Height)
        """
        if style is None:
                #print("return encode")
                return x
        else:
            m_batchsize,C,width ,height = x.size()
            proj_query  = self.query_conv(x).view(m_batchsize,-1,width*height).permute(0,2,1) # B X CX(N)
            proj_key =  self.key_conv(skel).view(m_batchsize,-1,width*height) # B X C x (*W*H)
            energy =  torch.bmm(proj_query,proj_key) # transpose check
            attention = self.softmax(energy) # BX (N) X (N) 
            proj_value = self.value_conv(skel).view(m_batchsize,-1,width*height) # B X C X N

            out = torch.bmm(proj_value,attention.permute(0,2,1) )
            out = out.view(m_batchsize,C,width,height)
            
            out = self.gamma*out + x
            
            return out, x.view(x.shape[0], -1)

class UnetSkipConnectionBlock(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, embedding_dim=128, norm_layer=nn.BatchNorm2d,
                 attn = None, use_dropout=False):
        """Construct a Unet submodule with skip connections.
        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            user_dropout (bool) -- if use dropout layers.
        """
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        self.innermost = innermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv, downnorm]
            up = [uprelu, upconv, nn.Tanh()]
            
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc + embedding_dim, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]

        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                up = up + [nn.Dropout(0.5)]

        self.submodule = submodule
        self.down = nn.Sequential(*down)
        self.up = nn.Sequential(*up)
        self.attn = attn
        

    def forward(self, x, style=None, skel=None):
        if self.innermost:
            enc = self.down(x)
            #print("layer 8")
            if style is None:
                #print("return encode")
                return self.submodule(enc)
            
            sub, encode = self.submodule(enc, style, skel)
            sub = torch.cat([style.view(style.shape[0], style.shape[1], 1, 1), sub], 1)
            dec = self.up(sub)
            return torch.cat([x, dec], 1), encode
        elif self.outermost:
            enc = self.down(x)
            #print("layer 1")
            if style is None:
                #print("return layer outer encode")
                return self.submodule(enc)
            sub, encode = self.submodule(enc, style, skel)
            dec = self.up(sub)
            return dec, encode
        else:  # add skip connections
            enc = self.down(x)
            #print("layer 2~7")
            if style is None:
                #print("return 2~7 encode")
                return self.submodule(enc)
            sub, encode = self.submodule(enc, style, skel)
            dec = self.up(sub)
            return torch.cat([x, dec], 1), encode
        

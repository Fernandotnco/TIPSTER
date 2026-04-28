import torch
import torch.nn as nn
import math



class ResBlock(nn.Module):
    def __init__(self, channels, activation=nn.SiLU, norm=nn.BatchNorm2d, dropout_p=0):
        super().__init__()
        self.act = activation()
        self.net = nn.Sequential(
            norm(channels),
            activation(),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False),
            activation(),  
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Dropout(p=dropout_p)
        )

    def forward(self, x):
        return self.net(x) + x


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, res_blocks=2,
                 activation=nn.SiLU, norm=nn.BatchNorm2d, dropout_p=0):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
        ]
        for _ in range(res_blocks):
            layers.append(ResBlock(out_channels, activation, norm, dropout_p=dropout_p))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class UpBlock(nn.Module):
    def __init__(
        self, in_channels, out_channels, res_blocks=2,
        activation=nn.SiLU, norm=nn.BatchNorm2d, final_activation=None
    ):
        super().__init__()
        layers = [
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
        ]
        for _ in range(res_blocks):
            layers.append(ResBlock(out_channels, activation, norm))
        if final_activation:
            layers.append(final_activation())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Unflatten(nn.Module):
    def __init__(self, channels, height, width):
        super().__init__()
        self.channels = channels
        self.height = height
        self.width = width

    def forward(self, x):
        return x.view(x.size(0), self.channels, self.height, self.width)

class Encoder(nn.Module):
    def __init__(
        self,
        in_channels=1,
        channels_list=[64, 128, 256, 512, 1024],
        res_blocks=[3, 3, 5, 3],
        embed_dim=256,
        activation=nn.SiLU,
        norm=nn.BatchNorm2d,
        latent_spatial_dim=None,
        final_activation=nn.Identity,
        dropout_p = 0,
        stride_on_stem=2,
        padding_on_stem=3,
        stem_kernel_size = 7,
        norm_on_stem = True
    ):
        super().__init__()
        assert latent_spatial_dim is not None, "Provide latent_spatial_dim=(C, H, W)"

        C, H, W = latent_spatial_dim

        # stem
        if norm_on_stem:
            layers = [
                nn.Conv2d(in_channels, channels_list[0], kernel_size=stem_kernel_size, stride=stride_on_stem, padding=padding_on_stem, bias=False),
                norm(channels_list[0]),
                activation(),
            ]
        else:
            layers = [
                nn.Conv2d(in_channels, channels_list[0], kernel_size=stem_kernel_size, stride=stride_on_stem, padding=padding_on_stem, bias=False),
                activation(),
            ]
        # down blocks
        for idx in range(len(channels_list) - 1):
            layers.append(DownBlock(
                channels_list[idx], channels_list[idx+1],
                res_blocks[idx], activation, norm, dropout_p=dropout_p
            ))
        self.conv = nn.Sequential(*layers)
        self.flatten = Flatten()
        self.fc = nn.Linear(C*W*H, embed_dim)
        self.final_activation = final_activation()
        self.embed_dim = embed_dim

    def forward(self, x):
        x = self.conv(x)
        #x = self.pool(x)
        x = self.flatten(x)
        z = self.fc(x)
        return self.final_activation(z)                            

class Decoder(nn.Module):
    def __init__(
        self,
        out_channels=1,
        channels_list=[64, 128, 256, 512, 1024],
        res_blocks=[3, 3, 5, 3],
        embed_dim=256,
        activation=nn.SiLU,
        norm=nn.BatchNorm2d,
        final_activation=None,
        latent_spatial_dim=None,
        output_shape = (128, 128),
        final_kernel_size=7
    ):
        super().__init__()
        assert latent_spatial_dim is not None, "Provide latent_spatial_dim=(C, H, W)"

        C, H, W = latent_spatial_dim
        self.fc = nn.Linear(embed_dim, C * H * W)
        self.unflatten = Unflatten(C, H, W)

        layers = []
        for idx in range(len(channels_list) - 1):
            layers.append(UpBlock(
                channels_list[idx], channels_list[idx+1],
                res_blocks[idx], activation, norm
            ))
        # final up + conv
        layers.append(nn.Upsample(size=output_shape, mode='nearest'))
        layers.append(nn.Conv2d(
            channels_list[-1], out_channels,
            kernel_size=final_kernel_size, stride=1, padding=final_kernel_size//2
        ))

        layers.append(final_activation())
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        x = self.fc(z)
        x = self.unflatten(x)
        return self.net(x)

class EncoderDecoder(nn.Module):
    def __init__(
        self,
        input_shape=(128, 128),
        in_channels=1,
        out_channels=1,
        embed_dim=256,
        channels_list=[64, 128, 256, 512, 1024],
        res_blocks=[3, 3, 5, 3],
        dropout_p = 0.2,
        stride_on_stem=2,
        padding_on_stem = 3,
        stem_kernel_size=7, 
        final_kernel_size=7,
        norm_on_stem=True,
        activation=nn.SiLU,
        norm=nn.BatchNorm2d,
        encoder_final_activation=nn.Identity,
        decoder_final_activation=nn.Identity
    ):
        super().__init__()

        self.embed_dim = embed_dim

        down_factor = 2 ** (len(res_blocks)) * stride_on_stem
        H, W = input_shape
        latent_h = int(math.floor(H / down_factor))
        latent_w =int(math.floor( W / down_factor))
        latent_dim = channels_list[-1]
        latent_spatial = (latent_dim, latent_h, latent_w)

        
        self.encoder = Encoder(
            in_channels=in_channels,
            channels_list=channels_list,
            res_blocks=res_blocks,
            embed_dim=embed_dim,
            activation=activation,
            norm=norm,
            latent_spatial_dim=latent_spatial,
            dropout_p=dropout_p,
            final_activation=encoder_final_activation,
            stride_on_stem=stride_on_stem,
            padding_on_stem = padding_on_stem,
            stem_kernel_size=stem_kernel_size,
            norm_on_stem=norm_on_stem
        )
        # compute latent spatial dims
        

        self.decoder = Decoder(
            out_channels=out_channels,
            channels_list=list(reversed(channels_list)),
            res_blocks=list(reversed(res_blocks)),
            embed_dim=embed_dim,
            activation=activation,
            norm=norm,
            final_activation=decoder_final_activation,
            latent_spatial_dim=latent_spatial,
            output_shape=input_shape,
            final_kernel_size=final_kernel_size
        )

    def forward(self, x, noise_mu=0):
        z = self.encoder(x)
        z += torch.randn_like(z)*noise_mu
        return self.decoder(z)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)



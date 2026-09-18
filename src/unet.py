"""Configurable 2D/3D U-Net for image-to-image synthesis.

The architecture (number of layers, filter counts, dropout, pooling/upsampling
type, etc.) is fully driven by the `encoder_layers` / `bottleneck` /
`decoder_layers` / `output_layer` sections of the YAML config, so the same
code builds both the 2D and 3D models used in this project.
"""

import torch
import torch.nn as nn


def get_conv_layer(dimensions):
    return nn.Conv2d if dimensions == 2 else nn.Conv3d


def get_batchnorm_layer(dimensions):
    return nn.BatchNorm2d if dimensions == 2 else nn.BatchNorm3d


def get_maxpool_layer(dimensions):
    return nn.MaxPool2d if dimensions == 2 else nn.MaxPool3d


def get_avgpool_layer(dimensions):
    return nn.AvgPool2d if dimensions == 2 else nn.AvgPool3d


def get_convtranspose_layer(dimensions):
    return nn.ConvTranspose2d if dimensions == 2 else nn.ConvTranspose3d


def get_dropout_layer(dimensions):
    return nn.Dropout2d if dimensions == 2 else nn.Dropout3d


def get_activation(activation_name):
    activations = {
        'relu': nn.ReLU(inplace=True),
        'leaky_relu': nn.LeakyReLU(0.2, inplace=True),
        'elu': nn.ELU(inplace=True),
        'gelu': nn.GELU(),
        'sigmoid': nn.Sigmoid(),
        'tanh': nn.Tanh(),
        'softmax': nn.Softmax(dim=1),
        'none': nn.Identity(),
    }
    return activations.get(activation_name.lower(), nn.ReLU(inplace=True))


class ConvBlock(nn.Module):
    """Conv -> BatchNorm -> Activation -> Dropout."""

    def __init__(self, in_ch, out_ch, kernel_size, stride, padding,
                 activation, batch_norm, dropout, dimensions):
        super().__init__()

        Conv = get_conv_layer(dimensions)
        BatchNorm = get_batchnorm_layer(dimensions)
        Dropout = get_dropout_layer(dimensions)

        layers = [Conv(in_ch, out_ch, kernel_size, stride, padding)]
        if batch_norm:
            layers.append(BatchNorm(out_ch))
        layers.append(get_activation(activation))
        if dropout > 0:
            layers.append(Dropout(dropout))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class EncoderBlock(nn.Module):
    """Conv blocks followed by pooling; returns the pooled output and the
    pre-pooling feature map (used as the skip connection)."""

    def __init__(self, in_ch, layer_config, dimensions):
        super().__init__()

        out_ch = layer_config['filters']
        num_convs = layer_config.get('num_convs', 2)

        conv_blocks = [
            ConvBlock(
                in_ch=in_ch if i == 0 else out_ch,
                out_ch=out_ch,
                kernel_size=layer_config['kernel_size'],
                stride=layer_config['stride'],
                padding=layer_config['padding'],
                activation=layer_config['activation'],
                batch_norm=layer_config['batch_norm'],
                dropout=layer_config['dropout'],
                dimensions=dimensions,
            )
            for i in range(num_convs)
        ]
        self.convs = nn.Sequential(*conv_blocks)

        pool_type = layer_config.get('pool_type', 'max')
        pool_size = layer_config['pool_size']
        pool_stride = layer_config['pool_stride']

        if pool_type == 'max':
            self.pool = get_maxpool_layer(dimensions)(kernel_size=pool_size, stride=pool_stride)
        elif pool_type == 'avg':
            self.pool = get_avgpool_layer(dimensions)(kernel_size=pool_size, stride=pool_stride)
        elif pool_type == 'conv':
            Conv = get_conv_layer(dimensions)
            self.pool = Conv(out_ch, out_ch, kernel_size=pool_size, stride=pool_stride, padding=0)

        self.out_channels = out_ch

    def forward(self, x):
        x = self.convs(x)
        skip = x
        x = self.pool(x)
        return x, skip


class BottleneckBlock(nn.Module):
    """Conv blocks with no pooling, sitting between encoder and decoder."""

    def __init__(self, in_ch, layer_config, dimensions):
        super().__init__()

        out_ch = layer_config['filters']
        num_convs = layer_config.get('num_convs', 2)

        conv_blocks = [
            ConvBlock(
                in_ch=in_ch if i == 0 else out_ch,
                out_ch=out_ch,
                kernel_size=layer_config['kernel_size'],
                stride=layer_config['stride'],
                padding=layer_config['padding'],
                activation=layer_config['activation'],
                batch_norm=layer_config['batch_norm'],
                dropout=layer_config['dropout'],
                dimensions=dimensions,
            )
            for i in range(num_convs)
        ]
        self.convs = nn.Sequential(*conv_blocks)
        self.out_channels = out_ch

    def forward(self, x):
        return self.convs(x)


class DecoderBlock(nn.Module):
    """Upsample -> concatenate skip connection -> conv blocks."""

    def __init__(self, in_ch, skip_ch, layer_config, dimensions):
        super().__init__()

        out_ch = layer_config['filters']
        upsample_type = layer_config.get('upsample_type', 'transpose')
        upsample_scale = layer_config.get('upsample_scale', 2)
        num_convs = layer_config.get('num_convs', 2)
        self.use_skip = layer_config.get('skip_connection', True)
        self.dimensions = dimensions

        if upsample_type == 'transpose':
            ConvTranspose = get_convtranspose_layer(dimensions)
            self.upsample = ConvTranspose(in_ch, in_ch, kernel_size=upsample_scale, stride=upsample_scale)
        else:
            mode = 'bilinear' if dimensions == 2 else 'trilinear'
            self.upsample = nn.Upsample(scale_factor=upsample_scale, mode=mode, align_corners=True)

        conv_in_ch = in_ch + skip_ch if self.use_skip else in_ch

        conv_blocks = [
            ConvBlock(
                in_ch=conv_in_ch if i == 0 else out_ch,
                out_ch=out_ch,
                kernel_size=layer_config['kernel_size'],
                stride=layer_config['stride'],
                padding=layer_config['padding'],
                activation=layer_config['activation'],
                batch_norm=layer_config['batch_norm'],
                dropout=layer_config['dropout'],
                dimensions=dimensions,
            )
            for i in range(num_convs)
        ]
        self.convs = nn.Sequential(*conv_blocks)
        self.out_channels = out_ch

    def forward(self, x, skip=None):
        x = self.upsample(x)

        if self.use_skip and skip is not None:
            if x.shape[2:] != skip.shape[2:]:
                x = self._match_size(x, skip)
            x = torch.cat([x, skip], dim=1)

        return self.convs(x)

    def _match_size(self, x, skip):
        """Pad x so its spatial size matches skip (rounding in pooling can
        otherwise leave off-by-one mismatches between encoder and decoder)."""
        if self.dimensions == 2:
            diff_h = skip.shape[2] - x.shape[2]
            diff_w = skip.shape[3] - x.shape[3]
            pad = [diff_w // 2, diff_w - diff_w // 2,
                   diff_h // 2, diff_h - diff_h // 2]
        else:
            diff_d = skip.shape[2] - x.shape[2]
            diff_h = skip.shape[3] - x.shape[3]
            diff_w = skip.shape[4] - x.shape[4]
            pad = [diff_w // 2, diff_w - diff_w // 2,
                   diff_h // 2, diff_h - diff_h // 2,
                   diff_d // 2, diff_d - diff_d // 2]
        return nn.functional.pad(x, pad)


class UNet(nn.Module):
    """Dynamic U-Net supporting 2D and 3D inputs, built from a layer-by-layer
    config so encoder depth, filter widths, and decoder skip connections are
    all configurable without touching this class."""

    def __init__(self, in_channels, out_channels, dimensions, input_shape,
                 encoder_layers, bottleneck, decoder_layers, output_layer):
        super().__init__()

        self.dimensions = dimensions
        self.input_shape = input_shape

        self.encoders = nn.ModuleList()
        self.skip_channels = []
        current_ch = in_channels

        for layer_config in encoder_layers:
            encoder = EncoderBlock(current_ch, layer_config, dimensions)
            self.encoders.append(encoder)
            self.skip_channels.append(encoder.out_channels)
            current_ch = encoder.out_channels

        self.bottleneck = BottleneckBlock(current_ch, bottleneck, dimensions)
        current_ch = self.bottleneck.out_channels

        self.decoders = nn.ModuleList()
        skip_channels_reversed = self.skip_channels[::-1]

        for i, layer_config in enumerate(decoder_layers):
            skip_ch = skip_channels_reversed[i] if layer_config.get('skip_connection', True) else 0
            decoder = DecoderBlock(current_ch, skip_ch, layer_config, dimensions)
            self.decoders.append(decoder)
            current_ch = decoder.out_channels

        Conv = get_conv_layer(dimensions)
        self.output_conv = Conv(
            current_ch,
            out_channels,
            kernel_size=output_layer['kernel_size'],
            stride=output_layer['stride'],
            padding=output_layer['padding'],
        )
        self.output_activation = get_activation(output_layer.get('activation', 'none'))

    def forward(self, x):
        input_size = x.shape[2:]

        skips = []
        for encoder in self.encoders:
            x, skip = encoder(x)
            skips.append(skip)

        x = self.bottleneck(x)

        skips = skips[::-1]
        for i, decoder in enumerate(self.decoders):
            skip = skips[i] if i < len(skips) else None
            x = decoder(x, skip)

        x = self.output_conv(x)
        x = self.output_activation(x)

        # Guard against small shape drift from pooling/upsampling rounding.
        if x.shape[2:] != input_size:
            mode = 'bilinear' if len(input_size) == 2 else 'trilinear'
            x = nn.functional.interpolate(x, size=input_size, mode=mode, align_corners=True)

        return x


def count_parameters(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def print_model_summary(model, input_shape):
    """Print architecture summary and run a dummy forward pass as a sanity check."""
    print("=" * 70)
    print("U-NET MODEL SUMMARY")
    print("=" * 70)
    print(f"Dimensions: {'2D' if model.dimensions == 2 else '3D'}")
    print(f"Input shape: {input_shape}")
    print("-" * 70)

    print("\nENCODER PATH:")
    for i, encoder in enumerate(model.encoders):
        print(f"  Encoder {i + 1}: {encoder.out_channels} filters")

    print(f"\nBOTTLENECK: {model.bottleneck.out_channels} filters")

    print("\nDECODER PATH:")
    for i, decoder in enumerate(model.decoders):
        skip_info = "+ skip" if decoder.use_skip else ""
        print(f"  Decoder {i + 1}: {decoder.out_channels} filters {skip_info}")

    print("-" * 70)

    trainable, total = count_parameters(model)
    print(f"\nTrainable parameters: {trainable:,}")
    print(f"Total parameters:     {total:,}")
    print(f"Model size:           {total * 4 / 1024 / 1024:.2f} MB (float32)")
    print("=" * 70)

    print("\nTesting forward pass...")
    device = next(model.parameters()).device
    dummy_input = torch.randn(input_shape).to(device)
    with torch.no_grad():
        output = model(dummy_input)
    print(f"Input shape:  {list(dummy_input.shape)}")
    print(f"Output shape: {list(output.shape)}")
    print("Forward pass successful!")
    print("=" * 70)


def build_unet(in_channels, out_channels, dimensions, input_shape,
               encoder_layers, bottleneck, decoder_layers, output_layer):
    """Build a U-Net from the config sections and print a summary + shape check."""
    model = UNet(
        in_channels=in_channels,
        out_channels=out_channels,
        dimensions=dimensions,
        input_shape=input_shape,
        encoder_layers=encoder_layers,
        bottleneck=bottleneck,
        decoder_layers=decoder_layers,
        output_layer=output_layer,
    )
    print_model_summary(model, input_shape)
    return model

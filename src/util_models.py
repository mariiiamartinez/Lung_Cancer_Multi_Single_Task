"""Model architectures for segmentation, classification and multi-task.

Supports two encoder families:
  - From scratch (UNet custom): 4-level encoder-decoder trained from
    random init. Two widths: init_features=32 (encoder 32-512,
    decoder 512-32) and init_features=64 (encoder 64-1024,
    decoder 1024-64).
  - Pretrained SMP: ResNet-18, ResNet-34 or ResNet-50 encoder
    (optionally with ImageNet weights) paired with the matching SMP
    UNet decoder.

Each architecture exposes three operating modes via use_segmentation
/ use_classification flags:

  - Segmentation only: encoder - decoder - segmentation head (1x1 conv).
  - Classification only: encoder - bottleneck - global pool - linear head
    (the decoder is not used).
  - Multi-task: encoder - decoder - segmentation head and classification
    head on the decoder output.
"""

from collections import OrderedDict
import inspect
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.decoders.unet.decoder import UnetDecoder as SMPUnetDecoder
import torch
import torch.nn as nn

_DECODER_CHANNELS = {
    'resnet18': (512, 256, 128, 64, 32),
    'resnet34': (512, 256, 128, 64, 32),
    'resnet50': (1024, 512, 256, 128, 64),
}


class UNet(nn.Module):
    """Shared UNet encoder-decoder body (from scratch).

    Builds the 4-level encoder, bottleneck and decoder with skip connections.
    Task heads and the forward pass live in the subclasses; this body is not
    meant to be used directly.
    """

    def __init__(self, in_channels=3, out_channels=1, init_features=32):
        super(UNet, self).__init__()

        self.features = init_features
        self.out_channels = out_channels

        # ---- Encoder ----
        self.encoder1 = UNet._block(in_channels, self.features, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = UNet._block(self.features, self.features * 2, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = UNet._block(self.features * 2, self.features * 4, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = UNet._block(self.features * 4, self.features * 8, name="enc4")
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # ---- Bottleneck ----
        self.bottleneck = UNet._block(self.features * 8, self.features * 16, name="bottleneck")

        # ---- Decoder ----
        self.upconv4 = nn.ConvTranspose2d(
            self.features * 16, self.features * 8, kernel_size=2, stride=2
        )
        self.decoder4 = UNet._block((self.features * 8) * 2, self.features * 8, name="dec4")
        self.upconv3 = nn.ConvTranspose2d(
            self.features * 8, self.features * 4, kernel_size=2, stride=2
        )
        self.decoder3 = UNet._block((self.features * 4) * 2, self.features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose2d(
            self.features * 4, self.features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = UNet._block((self.features * 2) * 2, self.features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose2d(
            self.features * 2, self.features, kernel_size=2, stride=2
        )
        self.decoder1 = UNet._block(self.features * 2, self.features, name="dec1")

    def _encode(self, x):
        """Encoder + bottleneck.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, C, H, W).

        Returns
        -------
        tuple[torch.Tensor, ...]
            ``(bottleneck, enc1, enc2, enc3, enc4)``. The encoder features
            are kept so the decoder can apply its skip connections.
        """
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))
        bottleneck = self.bottleneck(self.pool4(enc4))
        return bottleneck, enc1, enc2, enc3, enc4

    def _decode(self, bottleneck, enc1, enc2, enc3, enc4):
        """Decoder with skip connections.

        Parameters
        ----------
        bottleneck, enc1, enc2, enc3, enc4 : torch.Tensor
            Outputs of :meth:`_encode`.

        Returns
        -------
        torch.Tensor
            Decoder output ``dec1`` (B, features, H, W).
        """
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        return dec1

    @staticmethod
    def _block(in_channels, features, name):
        """Build a convolutional block of two Conv-BN-ReLU layers."""
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm2d(num_features=features)),
                    (name + "relu2", nn.ReLU(inplace=True)),
                ]
            )
        )


class UNetForSegmentationClassification(UNet):
    """UNet from scratch with optional segmentation and classification heads.

    Reuses the shared encoder-decoder body and adds either head
    independently via the ``use_segmentation`` / ``use_classification``
    flags.

    Segmentation head: a 1x1 conv on the decoder output (dec1) that
    produces the mask.

    Classification head:
      - Multi-task: mini-encoder block on dec1 (features to 128 channels),
        then global pooling and a linear layer.
      - Classification only: global pooling on the encoder bottleneck
        (features*16 channels) and a linear layer, no decoder involved.
    """

    def __init__(self, in_channels=3, out_channels=1, init_features=32, num_classes=2,
                 use_segmentation=True, use_classification=True):
        super(UNetForSegmentationClassification, self).__init__(in_channels, out_channels, init_features)

        self.num_classes = num_classes
        self.use_segmentation = use_segmentation
        self.use_classification = use_classification

        if self.use_segmentation:
            self.conv_segmentation_head = nn.Conv2d(
                in_channels=self.features, out_channels=self.out_channels, kernel_size=1
            )
        if self.use_classification:
            if self.use_segmentation:
                # Multi-task: classify on the decoder output (dec1), sharing
                # the learned features with the segmentation head.
                self.classification_decoder = nn.Sequential(
                    UNet._block(self.features, 128, name="cls_dec1"),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Flatten(),
                    nn.Dropout(p=0.25),
                    nn.Linear(128, 1 if self.num_classes == 2 else self.num_classes),
                )
            else:
                # Classification-only: global-pool the bottleneck features and
                # classify directly, with no decoder involved.
                self.classification_decoder = nn.Sequential(
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Flatten(),
                    nn.Dropout(p=0.25),
                    nn.Linear(self.features * 16, 1 if self.num_classes == 2 else self.num_classes),
                )

    def forward(self, x):
        """Forward pass producing segmentation mask and/or classification logits.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, C, H, W).

        Returns
        -------
        torch.Tensor or tuple[torch.Tensor, torch.Tensor]
            Segmentation mask (B, 1, H, W) and/or classification logits
            (B, num_classes), depending on the active heads.
        """
        bottleneck, enc1, enc2, enc3, enc4 = self._encode(x)

        if not self.use_segmentation and self.use_classification:
            return self.classification_decoder(bottleneck)

        dec1 = self._decode(bottleneck, enc1, enc2, enc3, enc4)

        outputs = []
        if self.use_segmentation:
            outputs.append(torch.sigmoid(self.conv_segmentation_head(dec1)))
        if self.use_classification:
            outputs.append(self.classification_decoder(dec1))

        if len(outputs) == 1:
            return outputs[0]
        return tuple(outputs)


class SMPUNetForSegmentationClassification(nn.Module):
    """SMP-backed UNet with optional segmentation and classification heads.

    Uses an SMP encoder (ResNet-18/34/50, optional ImageNet weights) with
    its matching UNet decoder. Heads are enabled independently via the
    ``use_segmentation`` / ``use_classification`` flags.

    Segmentation head: a 1x1 conv on the decoder output.

    Classification head:
      - Multi-task: mini-encoder block on the decoder output (decoder
        channels to 128), then global pooling and a linear layer.
      - Classification only: global pooling on the encoder bottleneck
        (out_channels[-1] channels) and a linear layer, no decoder involved.
    """

    def __init__(self, encoder_name='resnet34', encoder_weights=None, num_classes=2,
                 use_segmentation=True, use_classification=True):
        super(SMPUNetForSegmentationClassification, self).__init__()

        if encoder_name not in _DECODER_CHANNELS:
            raise ValueError(
                "Unsupported encoder for the multi-task head: %r. "
                "Supported encoders: %s." % (encoder_name, ', '.join(_DECODER_CHANNELS))
            )

        self.num_classes = num_classes
        self.use_segmentation = use_segmentation
        self.use_classification = use_classification
        self.encoder = smp.encoders.get_encoder(encoder_name, weights=encoder_weights, in_channels=3)
        decoder_channels = _DECODER_CHANNELS[encoder_name]
        if self.use_segmentation:
            # SMP UNet decoder with skip connections to the encoder stages.
            decoder_kwargs = dict(
                encoder_channels=self.encoder.out_channels,
                decoder_channels=decoder_channels,
                n_blocks=len(decoder_channels),
                use_batchnorm=True,
                center=False,
                attention_type=None,
            )
            accepted = inspect.signature(SMPUnetDecoder.__init__).parameters
            self.decoder = SMPUnetDecoder(**{
                key: value for key, value in decoder_kwargs.items() if key in accepted
            })
            self.segmentation_head = nn.Conv2d(decoder_channels[-1], 1, kernel_size=1)
        if self.use_classification:
            if self.use_segmentation:
                # Multi-task: classify on the decoder output, sharing the
                # learned features with the segmentation head.
                self.classification_decoder = nn.Sequential(
                    UNet._block(decoder_channels[-1], 128, name='cls_dec1'),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Flatten(),
                    nn.Dropout(p=0.25),
                    nn.Linear(128, 1 if num_classes == 2 else num_classes),
                )
            else:
                # Classification-only: global-pool the encoder bottleneck
                # (last encoder feature) and classify directly, with no decoder.
                self.classification_decoder = nn.Sequential(
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Flatten(),
                    nn.Dropout(p=0.25),
                    nn.Linear(self.encoder.out_channels[-1], 1 if num_classes == 2 else num_classes),
                )

    def forward(self, x):
        """Forward pass through the SMP-based model.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, 3, H, W).

        Returns
        -------
        torch.Tensor or tuple[torch.Tensor, torch.Tensor]
            Segmentation mask (B, 1, H, W) and/or classification logits
            (B, num_classes), depending on the active heads.
        """
        features = self.encoder(x)

        if not self.use_segmentation and self.use_classification:
            return self.classification_decoder(features[-1])

        # Old SMP decoders take ``features`` as one argument, newer ones as
        # ``*features``; detect it from the forward signature.
        decoder_forward = inspect.signature(SMPUnetDecoder.forward).parameters.values()
        if any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in decoder_forward):
            decoder_output = self.decoder(*features)
        else:
            decoder_output = self.decoder(features)

        outputs = []
        if self.use_segmentation:
            outputs.append(torch.sigmoid(self.segmentation_head(decoder_output)))
        if self.use_classification:
            outputs.append(self.classification_decoder(decoder_output))

        if len(outputs) == 1:
            return outputs[0]
        return tuple(outputs)
import torch.nn as nn
# import kornia ломает пайторч до версии 1.13.1 а текущая 1.0.0


class Sharp(nn.Module):
    def __init__(self, radius, percent, threshold):
        """
        PIL.ImageFilter.UnsharpMask
        PARAMETERS:
        radius – Blur Radius

        percent – Unsharp strength, in percent

        threshold – Threshold controls the minimum brightness change that will be sharpened


        Amount is listed as a percentage and controls the magnitude of each overshoot
        (how much darker and how much lighter the edge borders become). This can also be thought of as how much contrast
        is added at the edges. It does not affect the width of the edge rims.

        Radius affects the size of the edges to be enhanced or how wide the edge rims become, so a smaller radius
        enhances smaller-scale detail. Higher radius values can cause halos at the edges, a detectable faint light
        rim around objects. Fine detail needs a smaller radius. Radius and amount interact; reducing one allows
        more of the other.

        Threshold controls the minimal brightness change that will be sharpened or how far apart adjacent tonal values
        have to be before the filter does anything. This lack of action is important to prevent smooth areas from
        becoming speckled. The threshold setting can be used to sharpen more pronounced edges, while leaving subtler
        edges untouched. Low values should sharpen more because fewer areas are excluded. Higher threshold values
        exclude areas of lower contrast.

        """
        super(Sharp, self).__init__()
        self.radius = radius
        self.percent = percent
        self.threshold = threshold

    def forward(self, noised_and_cover):
        noised_image = noised_and_cover[0]
        return noised_and_cover

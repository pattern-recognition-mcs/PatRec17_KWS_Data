from xml.dom import minidom
import numpy as np
from scipy import ndimage
from matplotlib.path import Path
from matplotlib import pyplot as plt
from matplotlib import image

img = image.imread('images/270.jpg')

doc = minidom.parse('ground-truth/locations/270.svg')  # parseString also exists
masks={}
for path in doc.getElementsByTagName('path'):
    masks[path.getAttribute('id')] = path.getAttribute('d')
doc.unlink()

mask = {'270-01-01':np.array([[112.00, 170.00], [112.00, 230.00], [129.27, 231.50], [132.00, 230.00], [232.00, 230.00], [240.00, 238.00], [299.69, 148.25],  [192.00, 157.00]])}

pth = Path(mask['270-01-01'], closed=False)

nr, nc = img.shape
ygrid, xgrid = np.mgrid[:nr, :nc]
xypix = np.vstack((xgrid.ravel(), ygrid.ravel())).T


masked = np.ma.masked_array(img, ~mask)
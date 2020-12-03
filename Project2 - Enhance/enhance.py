#####################
## Noise Reduction ##
#####################

from skimage.filters.rank import mean, median

def noise_reduction_median(im, size):
    neighbourhood = np.ones((size, size))
    im_median = np.dstack([median(im[:, :, 0], selem=neighbourhood),
                          median(im[:, :, 1], selem=neighbourhood),
                          median(im[:, :, 2], selem=neighbourhood)])
    return im_median

def add_noise_3d(im, n):
    im2 = im.copy()
    size_x, size_y = im.shape[1], im.shape[0]
    rand_x = np.random.randint(0, size_x, size=(1, n))[0]
    rand_y = np.random.randint(0, size_y, size=(1, n))[0]
   # print(rand_x)
    for i in range(n):
        rand_color = np.random.randint(0, 256, size=(1,3))
        #print(i, rand_x[i], rand_y[i])
        im2[rand_y[i], rand_x[i]] = rand_color
        
    return im2

def add_noise_percentage(im, percentage):
    n = im.shape[0] * im.shape[1] * percentage
    n = round(n)
    return add_noise_3d(im, n)
	

################
## Histograms ##
################
def hist(im):
    h = np.zeros((256,))
    for i in range(256):
        h[i] = (im == i).sum()
    return h

def show_3d_hist(im):
    r, g, b = hist(im[:, :, 0]), hist(im[:, :, 1]), hist(im[:, :, 2])
    
    plt.figure()
    plt.subplot(1, 3, 1)
    plt.plot(r)
    plt.title('red')
    plt.subplot(1, 3, 2)
    plt.plot(g)
    plt.title('green')
    plt.subplot(1, 3, 3)
    plt.plot(b)
    plt.title('blue')
    plt.show()
	
	
def norm_cumulative_hist(im):
    h = np.zeros((256,))
    c = 0
    for i in range(256):
        c += (im == i).sum()
        h[i] = c
    h /= h.max()
    return h
	
def hsv_histogram(im):
    im_hsv = rgb2hsv(im)

    im_hsv = np.round(im_hsv*100)
    h, s, v = im_hsv[:, :, 0], im_hsv[:, :, 1], im_hsv[:, :, 2]
    print(v.sum(), im.shape[0], im.shape[1])
    size = im.shape[0] * im.shape[1]
    print(size)
    print(v.sum()/size)
    h_h = [(h==i).sum() for i in range(100)]
    h_s = [(s==i).sum() for i in range(100)]
    h_v = [(v==i).sum() for i in range(100)]


    plt.figure()

    plt.subplot(3, 1, 1)
    plt.plot(h_h)
    plt.title("hue")

    plt.subplot(3, 1, 2)
    plt.plot(h_s)
    plt.title("saturation")

    plt.subplot(3, 1, 3)
    plt.plot(h_v)
    plt.title("value")

    plt.show()
################
## Auto Level ##
################

def auto_level(im, Tmin, Tmax):
    lut = np.arange(0, 256)
    
    lut[Tmax:] = 255
    lut[:Tmin] = 0

    lut[Tmin:Tmax] = (lut[Tmin:Tmax] - Tmin)*(255/(Tmax-Tmin))

    im_str = lut[im]
    return im_str


def auto_level_3d(im, Tmin, Tmax):
    im2 = np.zeros(im.shape)
    for i in range(3):
        im2[:, :, i] = auto_level(im[:, :, i], Tmin[i], Tmax[i])
    return im2.astype('uint8')


def get_percentile_values(im, low_perc, high_perc):
    h = norm_cumulative_hist(im)
    low_val = 0
    high_val = 255
    for i in range(256):
        if h[i] >= low_perc:
            low_val = i
            break
            
    for i in range(255, -1, -1):
        if h[i] <= high_perc:
            high_val = i
            break
    
    return low_val, high_val


def auto_level_percentile(im, low_perc, high_perc):
    
    low_val, high_val = get_percentile_values(im, low_perc, high_perc)
    return auto_level(im, low_val, high_val)
    
def auto_level_percentile_3d(im, low_perc, high_perc):
    im2 = np.zeros(im.shape)
    
    for i in range(3):
        im2[:, :, i] = auto_level_percentile(im[:, :, i], low_perc[i], high_perc[i])
        
    return im2.astype('uint8')
	
	
######################
## Auto Level (HSV) ##
######################

def auto_level_hsv(im, Tmin, Tmax):
    lut = np.arange(0, 101)
    
    lut[Tmax:] = 100
    lut[:Tmin] = 0

    lut[Tmin:Tmax] = (lut[Tmin:Tmax] - Tmin)*(100/(Tmax-Tmin))

    im_str = lut[im]
    return im_str

######################
## Auto Level (gen) ##
######################	

def norm_cumulative_hist(im, max_val):
    h = np.zeros((max_val+1,))
    c = 0
    for i in range(max_val+1):
        c += (im == i).sum()
        h[i] = c
    h /= h.max()
    return h

def get_percentile_values(im, max_val, low_perc, high_perc):
    h = norm_cumulative_hist(im, max_val)
    #print(h)
    low_val = 0
    high_val = max_val
    for i in range(max_val+1):
        if h[i] >= low_perc:
            low_val = i
            break
            
    for i in range(max_val, -1, -1):
        if h[i] <= high_perc:
            high_val = i
            break
    
    return low_val, high_val

def auto_level(im, max_val, Tmin, Tmax):
    lut = np.arange(0, max_val+1)
    
    lut[Tmax:] = max_val
    lut[:Tmin] = 0

    lut[Tmin:Tmax] = (lut[Tmin:Tmax] - Tmin)*(max_val/(Tmax-Tmin))

    im_str = lut[im]
    return im_str


def auto_level_percentile(im, max_val, low_perc, high_perc):
    
    low_val, high_val = get_percentile_values(im, max_val, low_perc, high_perc)
    return auto_level(im, max_val, low_val, high_val)

def auto_level_percentile_3d(im, max_val, low_perc, high_perc):
    im2 = np.zeros(im.shape)
    
    for i in range(3):
        im2[:, :, i] = auto_level_percentile(im[:, :, i], max_val, low_perc[i], high_perc[i])
        
    return im2
	

################
## Saturation ##
################

def increase_saturation(im, factor):
    im_hsv = rgb2hsv(im)
    #print(im_hsv[:, :, 1].min(), im_hsv[:, :, 1].max())
    im_hsv[:, :, 1] *= factor
    im_hsv[im_hsv > 1] = 1

    return hsv2rgb(im_hsv)
	
	
######################
## Gamma correction ##
######################


def gamma_correction(im, gamma=1):
    im2 = (im.copy()).astype('float64')
    im2 /= 255
    im2 = im2**gamma
    im2 = np.round(im2*255)
    return im2.astype('uint8')

	
def auto_gamma(im):
    im_hsv = rgb2hsv(im)

    im_hsv = np.round(im_hsv*100)
    v =  im_hsv[:, :, 2]

    size = im.shape[0] * im.shape[1]
    
    ratio = v.sum()/size
    
    gamma = ratio/25
    
    return gamma_correction(im, gamma)
	


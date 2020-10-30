### CROP 

def watermark_transparent_crop(im, y, x, watermark, opacity=50):
    im = im.astype('int16')
    max_y =  y + watermark.shape[0]
    max_x =  x + watermark.shape[1]
    
    if im.shape[0] < max_y:
        max_y = im.shape[0]
        watermark = watermark[:im.shape[0]-y]  # crop the watermark
    if im.shape[1] < max_x:
        max_x = im.shape[1]
        watermark = watermark[:, :im.shape[1]-x]
        
    im2 = im.copy()
    im2[y:max_y, x:max_x][watermark>254] += np.array([opacity, opacity, opacity])
    im2[im2>255] = 255
    im2[im2<0] = 0
    return im2.astype(np.uint8)

def watermark_simple_crop(im, y, x, watermark):
    return watermark_transparent_crop(im, y, x, watermark, 255)
	
	
def watermark_average_crop(im, y, x, watermark, factor=0.5):
    im = im.astype('int16')
    max_y =  y + watermark.shape[0]
    max_x =  x + watermark.shape[1]
    
    
    if im.shape[0] < max_y:
        max_y = im.shape[0]
        watermark = watermark[:im.shape[0]-y]
    if im.shape[1] < max_x:
        max_x = im.shape[1]
        watermark = watermark[:, :im.shape[1]-x]
        
    im2 = im.copy()
    avg = (im2[y:max_y, x:max_x].sum())/((max_y-y)*(max_x-x)*3)
    
    if (avg>127):
        factor *= -1
    to_add = int(avg*factor)


    im2[y:max_y, x:max_x][watermark>254] += np.array([to_add, to_add, to_add])
    
    im2[im2>255] = 255
    im2[im2<0] = 0
    
    return im2.astype('uint8')
	
### LOOP

def watermark_transparent_loop(im, y, x, watermark, opacity=50):
    im = im.astype('int16')
    max_y =  y + watermark.shape[0]
    max_x =  x + watermark.shape[1]
    
    remainder_y = 0
    remainder_x = 0
    
    if im.shape[0] < max_y:
        remainder_y = max_y - im.shape[0]
        max_y = im.shape[0]
    if im.shape[1] < max_x:
        remainder_x = max_x - im.shape[1]
        max_x = im.shape[1]
        
    im2 = im.copy()
    
    im2[y:max_y, x:max_x][watermark[:max_y - y, :max_x - x]>254]           += np.array([opacity, opacity, opacity])
    im2[y:max_y, :remainder_x][watermark[:max_y - y, max_x - x:]>254]      += np.array([opacity, opacity, opacity])
    im2[:remainder_y, x:max_x][watermark[max_y - y:, :max_x - x]>254]      += np.array([opacity, opacity, opacity])
    im2[:remainder_y, :remainder_x][watermark[max_y - y:, max_x - x:]>254] += np.array([opacity, opacity, opacity])

    im2[im2>255] = 255
    im2[im2<0] = 0
    return im2.astype('uint8')


def watermark_simple_loop(im, y, x, watermark):  # we can again redefine simple mode using transparent mode
    return  watermark_transparent_loop(im, y, x, watermark, 255)
	
def watermark_average_loop(im, y, x, watermark, factor=0.5):
    im = im.astype('int16')
    max_y =  y + watermark.shape[0]
    max_x =  x + watermark.shape[1]
    
    remainder_y = 0
    remainder_x = 0
    
    if im.shape[0] < max_y:
        remainder_y = max_y - im.shape[0]
        max_y = im.shape[0]
    if im.shape[1] < max_x:
        remainder_x = max_x - im.shape[1]
        max_x = im.shape[1]
        
    im2 = im.copy()
    
        
    im3 = np.vstack((np.hstack((im2[y:max_y, x:max_x], im2[y:max_y, :remainder_x])),
                    np.hstack((im2[:remainder_y, x:max_x], im2[:remainder_y, :remainder_x]))))
    
    avg = (im3.sum())/((watermark.shape[0])*(watermark.shape[1])*3)
  
    if (avg>127):
        factor *= -1

    to_add = int(avg*factor)
    
    im2[y:max_y, x:max_x][watermark[:max_y - y, :max_x - x]>254]           += np.array([to_add, to_add, to_add])
    im2[y:max_y, :remainder_x][watermark[:max_y - y, max_x - x:]>254]      += np.array([to_add, to_add, to_add])
    im2[:remainder_y, x:max_x][watermark[max_y - y:, :max_x - x]>254]      += np.array([to_add, to_add, to_add])
    im2[:remainder_y, :remainder_x][watermark[max_y - y:, max_x - x:]>254] += np.array([to_add, to_add, to_add])
    
    im2[im2>255] = 255
    im2[im2<0] = 0
    
    return im2.astype('uint8')
	
### WRAPPER

def apply_watermark(im, y, x, watermark, mode="transparent", crop="crop", opacity=50, factor=0.5):
    if mode == "simple" and crop == "crop":
        return watermark_simple_crop(im, y, x, watermark)
    elif mode == "simple" and crop == "loop":
        return watermark_simple_loop(im, y, x, watermark)

    elif mode == "transparent" and crop == "crop":
        return watermark_transparent_crop(im, y, x, watermark, opacity)
    elif mode == "transparent" and crop == "loop":
        return watermark_transparent_loop(im, y, x, watermark, opacity)

    elif mode == "average" and crop == "crop":
        return watermark_average_crop(im, y, x, watermark, factor)
    elif mode == "average" and crop == "loop":
        return watermark_average_loop(im, y, x, watermark, factor)

    else:
        print("argument error")
    return None 
	
	
	
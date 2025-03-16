import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

class ImageHelper():

    @staticmethod
    def load_image(image_path:str, is_greyscale:bool=False):
        if is_greyscale:
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        else:
            image = cv2.imread(image_path)
        
        return image
    
    @staticmethod
    def show_image(image):
        if image is not None:
            cv2.imshow('image', image)
            
            while cv2.waitKey(1) != 27:
                pass
        else:
            print('Image is None')
        
        cv2.destroyAllWindows()

    @staticmethod
    def get_shape(image):
        return image.shape
    
    @staticmethod
    def get_type(image):
        return image.dtype
    
    @staticmethod
    def create_canvas(width:int, height:int, number_channel:int=3, color:int=255):
        canvas = np.zeros((height, width, number_channel), np.uint8)

        return canvas
    
    @staticmethod
    def update_image_color(image,channel:str=None, color:int=255):
        if channel is None:
            image[:,:,:] = color
        elif channel == 'r':
            image[:,:,2] = color
        elif channel == 'g':
            image[:,:,1] = color
        elif channel == 'b':
            image[:,:,0] = color

        return image
    
    @staticmethod
    def image_crop(image, start_x:int, end_x:int, start_y:int, end_y:int):
        return image[start_y:end_y, start_x:end_x]
    
    @staticmethod
    def convert_color_to_greyscale(image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    @staticmethod
    def convert_bgr_to_rgb(image):
        '''
        Convert image from BGR to RGB so we can display it using Matplotlib
        - In OpenCV, the default color space is BGR (Blue, Green, Red)
        - In Matplotlib, the default color space is RGB (Red, Green, Blue)
        '''
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    @staticmethod
    def plot_image(image, heigh:int=16, width:int=6, title:str='Image'):
        '''
        Plot image by using Matplotlib

        Parameters:
        - image: image that you want to display
        - heigh: height of the image
        - width: width of the image
        - title: title of the image
        '''
        plt.figure(figsize=(heigh, width))
        plt.title(title), plt.xticks([]), plt.yticks([])
        plt.imshow(image)

    @staticmethod
    def reduce_noise_medianblur(image, blur_level:float=1):
        return cv2.medianBlur(image, blur_level)
    
    @staticmethod
    def reduce_noise_fastNlMeands(image):
        return cv2.fastNlMeansDenoisingColored(image , None, 10, 10, 7, 15)
    
    @staticmethod
    def thesholding(image):
        return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    
    @staticmethod
    def upscale_image(image):
        return cv2.resize(image, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    
    @staticmethod
    def normalize_image(image):
        norm_img = np.zeros((image.shape[0], image.shape[1]))
        image_normalize = cv2.normalize(image, norm_img, 0, 255, cv2.NORM_MINMAX)
        return image_normalize
    
    @staticmethod
    def adjust_contrast(image):
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        return(clahe.apply(image))
    
    @staticmethod
    def crop_white_borders(image, tolerance=10):

        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Invert the image (so text becomes white and background becomes black)
        _, thresh = cv2.threshold(gray, 255 - tolerance, 255, cv2.THRESH_BINARY_INV)

        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # If no contours found, return original
        if not contours:
            return image

        # Get bounding box around the content
        x, y, w, h = cv2.boundingRect(np.concatenate(contours))

        # Crop the image
        cropped_image = image[y:y+h, x:x+w]

        return cropped_image
    
    @staticmethod
    def deskew_image(image):
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Threshold image
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Detect coordinates of non-zero pixels
        coords = np.column_stack(np.where(thresh > 0))

        # Get rotation angle
        angle = cv2.minAreaRect(coords)[-1]
        print(angle)

        # Adjust angle
        if angle > -45:
            angle = 90 - angle
        else:
            angle = -angle

        # Rotate image to correct angle
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

        return rotated, angle
        

    # -- Deprecate
    
    # IMAGE_SIZE = 1800
    # BINARY_THREHOLD = 180

    # @staticmethod
    # def process_image_for_ocr(image):
    #     # TODO : Implement using opencv
    #     image_new = ImageHelper.remove_noise_and_smooth(image)
    #     return image_new

    # def set_image_dpi(file_path):
    #     im = Image.open(file_path)
    #     length_x, width_y = im.size
    #     factor = max(1, int(IMAGE_SIZE / length_x))
    #     size = factor * length_x, factor * width_y
    #     # size = (1800, 1800)
    #     im_resized = im.resize(size, Image.ANTIALIAS)
    #     temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
    #     temp_filename = temp_file.name
    #     im_resized.save(temp_filename, dpi=(300, 300))
    #     return temp_filename

    # def image_smoothening(img):
    #     ret1, th1 = cv2.threshold(img, ImageHelper.BINARY_THREHOLD, 255, cv2.THRESH_BINARY)
    #     ret2, th2 = cv2.threshold(th1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    #     blur = cv2.GaussianBlur(th2, (1, 1), 0)
    #     ret3, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    #     return th3

    # def remove_noise_and_smooth(image):
    #     filtered = cv2.adaptiveThreshold(image.astype(np.uint8), 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 41,
    #                                     3)
    #     kernel = np.ones((1, 1), np.uint8)
    #     opening = cv2.morphologyEx(filtered, cv2.MORPH_OPEN, kernel)
    #     closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
    #     image = ImageHelper.image_smoothening(image)
    #     or_image = cv2.bitwise_or(image, closing)
    #     return or_image


